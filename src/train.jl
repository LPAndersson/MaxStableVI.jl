import Flux
import Random
import StatsBase

using StatsFuns: logsumexp
using Statistics: mean

function train!(rng::Random.AbstractRNG,
    model::AbstractMaxStableModel,
    guide::AbstractGuide;
    data::Vector{Matrix{Float64}},
    epochs::Int64 = 10,
    M::Int64 = 1,
    guideopt::Flux.Optimise.AbstractOptimiser,
    modelopt::Flux.Optimise.AbstractOptimiser,
    printing::Bool = true
    )

    observations = data[1]
    coordinates = data[2]

    guide_opt_state = Flux.setup(guideopt,guide)
    model_opt_state = Flux.setup(modelopt,model)

    (n, d) = size(observations)
        
    modelHist = Vector{AbstractMaxStableModel}(undef,0)
    guideHist = Vector{AbstractGuide}(undef,0)
    
    elboHist = Vector{Float64}(undef,0)  

    guideSamples = [[[1]] for _ in 1:M]

    guideValues = Vector{Float64}(undef, M)
    guideGrads = Vector{Vector{Array{Float64}}}(undef, M)

    pqgradLogq = Vector{Vector{Array{Float64}}}(undef, M)
    pqgradLogp = Array{Vector{Array{Float64}}}(undef, M)
    
    modelValues = Vector{Float64}(undef, M)
    modelGrads = Vector{Vector{Array{Float64}}}(undef, M)

    c = 0

    batchOrder = StatsBase.sample(rng, 1:n, n, replace = false)
  
    for epoch in 1:epochs

        elboEstimate = 0.0

        for obsIdx in batchOrder
            Threads.@threads for m in 1:M

                (guideValues[m], g) = 
                    Flux.withgradient( 
                        g -> sample(
                            g, 
                            observations[obsIdx,:],
                            coordinates,
                            guideSamples[m]
                            ), 
                        guide
                        )

                guideGrads[m] = collect(g[1])

                (modelValues[m], g) = 
                Flux.withgradient( 
                    model -> condLogLikelihood(
                        model, 
                        observations[obsIdx,:], 
                        coordinates, 
                        guideSamples[m]
                        ), 
                    model
                    )

                modelGrads[m] = collect(g[1])

                pqgradLogq[m] = exp(modelValues[m] - guideValues[m]) .* guideGrads[m]
                pqgradLogp[m] = exp(modelValues[m] - guideValues[m]) .* modelGrads[m]
        
            end
    
            guideGradSum = reduce(.+, guideGrads)
        
            log_pqsum = logsumexp(modelValues .- guideValues)
                
            modelStep =  -exp(-log_pqsum) .* reduce(.+, pqgradLogp)
            guideStep =  -(-exp(-log_pqsum) .* reduce(.+, pqgradLogq) .+ (log_pqsum - c) .* guideGradSum)

            model_opt_state, model = Flux.update!(model_opt_state, model, (; zip(fieldnames(typeof(model)), modelStep)...))
            guide_opt_state, guide = Flux.update!(guide_opt_state, guide, (; zip(fieldnames(typeof(guide)), guideStep)...))
            
            clamp!(model)
            clamp!(guide)
        
            elboEstimate += log_pqsum - log(M)

            c = 0.99 * c + (1-0.99) * log_pqsum

        end

        push!(modelHist, deepcopy(model))
        push!(guideHist, deepcopy(guide))
        push!(elboHist, elboEstimate)

        if printing 
            println("Epoch ", epoch,"/", epochs, " Elbo " , elboEstimate)
        end
  
    end
≈
    return Dict([
        ("model", modelHist),
        ("guide", guideHist),
        ("elbo", elboHist)
    ])
end

train!(model::AbstractMaxStableModel,
    guide::AbstractGuide;
    data::Vector{Matrix{Float64}},
    epochs::Int64 = 10,
    M::Int64 = 1,
    guideopt::Flux.Optimise.AbstractOptimiser,
    modelopt::Flux.Optimise.AbstractOptimiser,
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

  