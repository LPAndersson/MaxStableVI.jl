import Random
import Combinatorics: bellnum, SetPartitions
import StatsFuns: logsumexp

using FLoops

function elboMC(
    rng::Random.AbstractRNG,
    model::AbstractMaxStableModel, 
    guide::AbstractGuide;
    data::Vector{Matrix{Float64}},
    numOfSamples::Int64,
    M::Int64 = 1)

    observations = data[1]
    coordinates = data[2]

    (n, d) = size(observations)

    elbo = 0.0
      
    @floop for countSample in 1:numOfSamples
        guideSample::Vector{Vector{Int64}} = [[]]
        logSum = 0.0

        pq = Vector{Float64}(undef, M)
        
        for countObservation in 1:n
            for m in 1:M
                guideLogL = 
                    sample(
                        rng,
                        guide, 
                        observations[countObservation,:], 
                        coordinates, 
                        guideSample
                    )

                modelLogL = condLogLikelihood(model, observations[countObservation,:], coordinates, guideSample)
                pq[m] = modelLogL - guideLogL
            end
            logSum += logsumexp(pq) - log(M)
        end

        @reduce(elbo += logSum)
    end
  
    return elbo/numOfSamples

end
elboMC(
    model::AbstractMaxStableModel, 
    guide::AbstractGuide;
    data::Vector{Matrix{Float64}},
    numOfSamples::Int64,
    M::Int64 = 1
    ) = elboMC(
        Random.default_rng(), 
        model, 
        guide, 
        data = data, 
        numOfSamples = numOfSamples,
        M = M
        )

function logLikelihoodIS(
    rng::Random.AbstractRNG,
    model::AbstractMaxStableModel, 
    guide::AbstractGuide;
    data::Vector{Matrix{Float64}},
    numOfSamples::Int64)

    observations = data[1]
    coordinates = data[2]

    (n, d) = size(observations)

    for countObservation in 1:n
        guideLogLikelihood = Vector{Float64}(undef,numOfSamples)
        modelLogLikelihood = Vector{Float64}(undef,numOfSamples)
        guideSample = Vector{Vector{Int64}}(undef,0)
        @floop for countSamples in 1:numOfSamples
            guideLogLikelihood[countSamples] = sample(rng, guide, observations[countObservation,:], coordinates, guideSample)
            modelLogLikelihood[countSamples] = condLogLikelihood(model, observations[countObservation,:], coordinates, guideSample)
        end
        @reduce( logLikelhood += logsumexp(modelLogLikelihood .- guideLogLikelihood) - log(numOfSamples) )
    end
    
    return logLikelhood 
  
end
logLikelihoodIS(
    model::AbstractMaxStableModel, 
    guide::AbstractGuide;
    data::Vector{Matrix{Float64}},
    numOfSamples::Int64
    ) = logLikelihoodIS(
        Random.default_rng(),
        model, 
        guide, 
        data = data,
        numOfSamples = numOfSamples
        )
        
function loglikelihoodEnumerate(
    model::AbstractMaxStableModel, 
    data::Vector{Matrix{Float64}})

    observations, coordinates = data

    (n, d) = size(observations)

    @floop for countObservation in 1:n
        conditionalLoglikehood = Array{Float64}(undef, bellnum(d))
        partitionIterator = SetPartitions(1:d)
        countPartitions = 0
        for partition in partitionIterator
            countPartitions += 1
            conditionalLoglikehood[countPartitions] = condLogLikelihood(model, observations[countObservation,:], coordinates,  partition)
        end
        @reduce(loglikelihood += logsumexp(conditionalLoglikehood) )
    end
    return loglikelihood
end

function compositeLogLikelihood(
    model::AbstractMaxStableModel,
    data::Vector{Matrix{Float64}},
    degree::Int64
    ) 

    observations, coordinates = data
    n, d = size(observations)
    logl = 0

    for s in powerset(1:d,degree,degree)
        logl = logl + loglikelihoodEnumerate(
            model,
            [observations[:,s], coordinates[s,:]]
        )
    end
    return logl
  end