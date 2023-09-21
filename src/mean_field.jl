import Random
import Zygote
import StatsFuns
import Statistics
import Distributions

import Flux: @functor

mutable struct MeanField <: AbstractGuide
    #customers are in rows and tables in columns
    weights::Matrix{Float64}
end

MeanField(; weights::Matrix{Float64}) = MeanField(weights)

@functor MeanField

function clamp!(guide::MeanField)

    guide.weights = guide.weights .- Statistics.mean(guide.weights, dims = 2)

    return Nothing

end

function sample(
    rng::Random.AbstractRNG,
    guide::MeanField, 
    observation::Vector{Float64},
    coordinate::Matrix{Float64},
    partition::Vector{Vector{Int64}},
    obsNum::Int64
    )
  
    weights = guide.weights

    d = length(observation)

    partition_local = Vector{Vector{Int64}}(undef,d)

    logLikelihood = 0.0

    for customerNumber in 1:d
  
        tableProbs = StatsFuns.softmax(weights[customerNumber,:])
        chosenTable = Random.rand(rng, Distributions.Categorical(tableProbs))

        if isassigned(partition_local,chosenTable)
            Zygote.@ignore push!(partition_local[chosenTable], customerNumber)
        else
            Zygote.@ignore partition_local[chosenTable] = [customerNumber]
        end

        logLikelihood += log(tableProbs[chosenTable])
    end

    Zygote.ignore() do 
        copy!(partition, partition_local[[isassigned(partition_local,i) for i in 1:length(partition_local)]])
    end
  
    return logLikelihood # return assignment labels and permuted indices
end

sample(
    guide::MeanField, 
    observation::Vector{Float64},
    coordinate::Matrix{Float64},
    partition::Vector{Vector{Int64}},
    obsNum::Int64
    ) = sample(
    Random.default_rng(),
    guide, 
    observation,
    coordinate,
    partition,
    obsNum
    )