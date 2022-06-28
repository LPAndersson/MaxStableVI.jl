import Flux
import Zygote
import Random
import StatsBase

using StatsFuns: logsumexp
using Statistics: mean

function train!(rng::Random.AbstractRNG,
    model::AbstractMaxStableModel, 
    guide::AbstractGuide, 
    data::Vector{Matrix{Float64}};
    epochs::Int64 = 10,
    numSamplesIn::Int64 = 1,
    guideopt::Flux.Optimise.AbstractOptimiser,
    modelopt::Flux.Optimise.AbstractOptimiser
    )
    
    observations = data[1]
    coordinates = data[2]

    (n, d) = size(observations)

    #traceStep  = 10

    movingAvgLength = 50
    movingAvg = zeros(Float64,movingAvgLength)

    #modelParamHist = Vector{Zygote.Params}(undef,0)
    #guideParamHist = Vector{Zygote.Params}(undef,0)
    
    modelParamHist = Vector{Vector{Float64}}(undef,0)
    guideParamHist = Vector{Vector{Float64}}(undef,0)
    
    elboHist = Vector{Float64}(undef,0)  

    guideSamples = [[[1]] for _ in 1:numSamplesIn]

    guideValues = Vector{Float64}(undef, numSamplesIn)
    guideGrads = Vector{Zygote.Grads}(undef, numSamplesIn)

    pqgradLogq = Array{Zygote.Grads}(undef, numSamplesIn)
    pqgradLogp = Array{Zygote.Grads}(undef, numSamplesIn)
    
    modelValues = Vector{Float64}(undef, numSamplesIn)
    modelGrads = Vector{Zygote.Grads}(undef, numSamplesIn)
        
    guideParams = Flux.params(guide)
    modelParams = Flux.params(model)

    batchOrder = StatsBase.sample(rng, 1:n, n, replace = false)
  
    for epoch in 1:epochs

        println("Epoch ", epoch,"/", epochs)

        elboEstimate = 0.0

        for obsIdx in batchOrder
            Threads.@threads for m in 1:numSamplesIn
    
                (guideValues[m], guideGrads[m]) = 
                    Zygote.withgradient( 
                        () -> sample(
                                guide, 
                                observations[obsIdx,:],
                                coordinates,
                                guideSamples[m]
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

            c = elboEstimate
        
            log_pqsum = logsumexp(modelValues .- guideValues)
                
            modelStep =  exp(-log_pqsum) .* reduce(.+, pqgradLogp)
            guideStep = -exp(-log_pqsum) .* reduce(.+, pqgradLogq) .+ (log_pqsum-c) .* guideGradSum
        
            Flux.update!(modelopt, modelParams, (-1).* modelStep)
            Flux.update!(guideopt, guideParams, (-1).* guideStep)   
            
            clamp!(model)
            clamp!(guide)    
        
            elboEstimate += logsumexp(modelValues .- guideValues) - log(numSamplesIn)

        end

        c = 0.0#elboEstimate

        #push!(modelParamHist, deepcopy(modelParams))
        push!(modelParamHist, getindex.(modelParams[:],1))
        push!(guideParamHist, getindex.(guideParams[:],1))
        push!(elboHist, elboEstimate)
  
    end
≈
    return Dict([
        ("model", modelParamHist),
        ("guide", guideParamHist),
        ("elbo", elboHist)
    ])
end

train!(model::AbstractMaxStableModel, 
    guide::AbstractGuide, 
    data::Vector{Matrix{Float64}};
    epochs::Int64 = 10,
    numSamplesIn::Int64 = 1,
    guideopt::Flux.Optimise.AbstractOptimiser,
    modelopt::Flux.Optimise.AbstractOptimiser,
    batchsize::Int64 = 1
    ) = train!(
        Random.default_rng(),
        model, 
        guide, 
        data;
        epochs,
        numSamplesIn,
        guideopt,
        modelopt
        )

  