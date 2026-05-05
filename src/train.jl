import Flux
import Zygote
import Random
import StatsBase

using StatsFuns: logsumexp
using Statistics: mean

# ---------------------------------------------------------------------------
# Helpers for gradient-tree algebra (works with any Functors-compatible tree,
# including nested Flux.Chain gradients).
# ---------------------------------------------------------------------------

# Scale every Array leaf in gradient tree g by scalar c.
_scale_grad(c::Real, g) = Flux.fmap(x -> x isa AbstractArray ? c .* x : x, g)

# Element-wise addition of two gradient trees with matching structure.
_add_grads(a, b) = Flux.fmap((x, y) -> x isa AbstractArray ? x .+ y : x, a, b)

# ---------------------------------------------------------------------------

function train!(rng::Random.AbstractRNG,
    model::AbstractMaxStableModel,
    guide::AbstractGuide;
    data::Vector{Matrix{Float64}},
    epochs::Int64 = 10,
    M::Int64 = 1,
    guideopt,
    modelopt,
    printing::Bool = true
    )

    observations = data[1]
    coordinates = data[2]

    (n, d) = size(observations)

    modelHist = Vector{typeof(model)}(undef, epochs)
    guideHist = Vector{typeof(guide)}(undef, epochs)
    elboHist  = Vector{Float64}(undef, epochs)

    # One partition buffer per importance sample slot.
    guideSamples = [[[1]] for _ in 1:M]

    # Set up explicit optimizer states (Flux 0.14+ / Optimisers.jl API).
    guide_state = Flux.setup(guideopt, guide)
    model_state = Flux.setup(modelopt, model)

    batchOrder = StatsBase.sample(rng, 1:n, n, replace = false)

    for epoch in 1:epochs

        elboEstimate = 0.0

        for obsIdx in batchOrder

            guideValues   = Vector{Float64}(undef, M)
            modelValues   = Vector{Float64}(undef, M)
            guide_raw_gs  = Vector{Any}(undef, M)   # raw ∇_φ log q
            pqgradLogq    = Vector{Any}(undef, M)   # w_m · ∇_φ log q_m
            pqgradLogp    = Vector{Any}(undef, M)   # w_m · ∇_θ log p_m

            for m in 1:M
                # --- guide gradient -----------------------------------------
                # sample() writes the drawn partition into guideSamples[m]
                # as a side-effect inside Zygote.ignore(), then returns log q.
                guideValues[m], gs_g = Flux.withgradient(guide) do g
                    sample(g, observations[obsIdx,:], coordinates,
                           guideSamples[m], obsIdx)
                end
                guide_raw_gs[m] = gs_g[1]

                # --- model gradient (partition fixed from guide sample above) -
                modelValues[m], gs_m = Flux.withgradient(model) do mo
                    condLogLikelihood(mo, observations[obsIdx,:],
                                      coordinates, guideSamples[m])
                end

                w = exp(modelValues[m] - guideValues[m])
                pqgradLogq[m] = _scale_grad(w, gs_g[1])
                pqgradLogp[m] = _scale_grad(w, gs_m[1])
            end

            log_pqsum = logsumexp(modelValues .- guideValues)

            guide_grad_sum = reduce(_add_grads, guide_raw_gs)

            # Importance-weighted gradient directions (pre-negation).
            model_update = _scale_grad(
                exp(-log_pqsum),
                reduce(_add_grads, pqgradLogp)
            )
            guide_update = _add_grads(
                _scale_grad(-exp(-log_pqsum), reduce(_add_grads, pqgradLogq)),
                _scale_grad(log_pqsum, guide_grad_sum)
            )

            # Flux.update! performs param -= lr * grad (gradient descent).
            # We want gradient *ascent* on the ELBO, so negate the updates.
            Flux.update!(model_state, model, _scale_grad(-1.0, model_update))
            Flux.update!(guide_state, guide, _scale_grad(-1.0, guide_update))

            clamp!(model)
            clamp!(guide)

            elboEstimate += log_pqsum - log(M)
        end

        modelHist[epoch] = deepcopy(model)
        guideHist[epoch] = deepcopy(guide)
        elboHist[epoch]  = elboEstimate

        if printing
            println("Epoch ", epoch, "/", epochs, " Elbo ", elboEstimate)
        end

    end

    return Dict([
        ("model", modelHist),
        ("guide", guideHist),
        ("elbo",  elboHist)
    ])
end

train!(model::AbstractMaxStableModel,
    guide::AbstractGuide;
    data::Vector{Matrix{Float64}},
    epochs::Int64 = 10,
    M::Int64 = 1,
    guideopt,
    modelopt,
    printing::Bool = true
    ) = train!(
        Random.default_rng(),
        model,
        guide;
        data,
        epochs,
        M,
        guideopt,
        modelopt,
        printing
    )
