import Random
import Flux
import Zygote
import StatsFuns
import Statistics
import Distributions
import SliceMap

import Flux: @functor

 mutable struct MultinomialClustering <: AbstractGuide
    h::Vector{Float32}
end

MultinomialClustering(;D::Integer) = 
MultinomialClustering(
    rand(Float32,(D))
)

@functor MultinomialClustering

function clamp!(guide::MultinomialClustering)

    return Nothing

end

function sample(
    rng::Random.AbstractRNG,
    guide::MultinomialClustering, 
    observation::Vector{Float64},
    coordinate::Matrix{Float64},
    partition::Vector{Vector{Int64}},
    obsNum::Int64
    )
  
    h = guide.h

    N = length(h)

    p = Flux.softmax(h)
    numOfTables = Random.rand(rng, Distributions.Categorical(p))
    logLikelihood = log(p[numOfTables])

    Zygote.ignore() do 
        c = [collect(1:numOfTables); Distributions.wsample(1:numOfTables,ones(numOfTables),N-numOfTables)]
        reorder = Random.randperm(N)
        partition_local = [[] for _ in 1:maximum(c)]
        for (index, value) in enumerate(c)
            push!(partition_local[value], reorder[index])
        end
        copy!(partition, partition_local)
    end

    return logLikelihood # return assignment labels and permuted indices
end

sample(
    guide::MultinomialClustering, 
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


function logLikelihood(
    guide::MultinomialClustering, 
    observation::Vector{Float64},
    coordinate::Matrix{Float64},
    partition::Vector{Vector{Int64}},
    obsNum::Int64
    )
    
    h = guide.H

    return log(Flux.softmax(h)[length(partition_local)])
end