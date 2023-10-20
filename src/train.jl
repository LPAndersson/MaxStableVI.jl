import Flux
import Zygote
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

    (n, d) = size(observations)
        
    modelHist = Vector{typeof(model)}(undef,epochs)
    guideHist = Vector{typeof(guide)}(undef,epochs)
    
    elboHist = Vector{Float64}(undef,epochs)  

    guideSamples = [[[1]] for _ in 1:M]

    guideValues = Vector{Float64}(undef, M)
    guideGrads = Vector{Zygote.Grads}(undef, M)

    pqgradLogq = Array{Zygote.Grads}(undef, M)
    pqgradLogp = Array{Zygote.Grads}(undef, M)
    
    modelValues = Vector{Float64}(undef, M)
    modelGrads = Vector{Zygote.Grads}(undef, M)
        
    guideParams = Flux.params(guide)
    modelParams = Flux.params(model)

    #c = 0

    batchOrder = StatsBase.sample(rng, 1:n, n, replace = false)
  
    for epoch in 1:epochs

        elboEstimate = 0.0

        for obsIdx in batchOrder
            #Threads.@threads 
            for m in 1:M
    
                (guideValues[m], guideGrads[m]) = 
                    Zygote.withgradient( 
                        () -> sample(
                                guide, 
                                observations[obsIdx,:],
                                coordinates,
                                guideSamples[m],
                                obsIdx
                                ), 
                        guideParams
                        )

                (modelValues[m], modelGrads[m]) = 
                    Zygote.withgradient( 
                        () -> condLogLikelihood(
                            model, 
                            observations[obsIdx,:], 
                            coordinates, 
                            guideSamples[m]
                            ), 
                        modelParams
                        )

                pqgradLogq[m] = exp(modelValues[m] - guideValues[m]) .* guideGrads[m]
                pqgradLogp[m] = exp(modelValues[m] - guideValues[m]) .* modelGrads[m]
        
            end
    
            guideGradSum = reduce(.+, guideGrads)

            #c = elboEstimate
        
            log_pqsum = logsumexp(modelValues .- guideValues)
                
            modelStep =  exp(-log_pqsum) .* reduce(.+, pqgradLogp)
            guideStep =  -exp(-log_pqsum) .* reduce(.+, pqgradLogq) .+ log_pqsum .* guideGradSum

            Flux.update!(modelopt, modelParams, (-1).* modelStep)
            Flux.update!(guideopt, guideParams, (-1).* guideStep)   
            
            clamp!(model)
            clamp!(guide)

            elboEstimate += log_pqsum - log(M)

            #c = 0.99 * c + (1-0.99) * log_pqsum

        end

        modelHist[epoch] = deepcopy(model)
        guideHist[epoch] = deepcopy(guide)
        elboHist[epoch] = elboEstimate

#=         push!(modelParamHist, getindex.(modelParams[:],1))
        push!(guideParamHist, getindex.(guideParams[:],1))
        push!(elboHist, elboEstimate)
 =#
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

  