import Random
import Zygote
import StatsFuns
import Statistics
import Distributions

import Flux: @functor

mutable struct SimpleClustering <: AbstractGuide
    #customers are in rows and tables in columns
    h::Vector{Float64}
    weights::Vector{Float64}
end

SimpleClustering(; h::Matrix{Float64},  weights::Matrix{Float64}) = SimpleClustering(h,weights)

@functor SimpleClustering

function clamp!(guide::SimpleClustering)

    return Nothing

end

function sample(
    rng::Random.AbstractRNG,
    guide::SimpleClustering, 
    observation::Vector{Float64},
    coordinate::Matrix{Float64},
    partition::Vector{Vector{Int64}},
    obsNum::Int64
    )
  
    h = guide.h
    u = h
    weights = guide.weights

    N = length(observation)

    c = Vector{Int64}(undef,N)
    H = Vector{Float64}(undef,N)
    logLikelihood = 0.0

    U = sum(u[2:end])
    H[1] = h[1]
    G = H[1]
    K = 1
    c[1] = 1

    for n in 2:N
        U = U - u[n]
        H[K+1] = 0
        for k in 1:(K+1)

            q = map((x) -> exp(weights[1] * G + h + weights[2] * U))

            p[k] = exp(weights[1]*G + weights[2]*Q + weights[3]*h[n], )
            G = G - h[n]
        end
        p = p./sum(p[1:(K+1)])
        Zygote.@ignore c[n] = Random.rand(rng, Distributions.Categorical(p[1:(K+1)]))
        logLikelihood += log(copy(p[c[n]]))
        if c[n] == K+1
            K = K + 1
        end
        G = G + h[n]
    end
  
    Zygote.ignore() do 
        partition_local = [[] for _ in 1:maximum(c)]
        for (index, value) in enumerate(c)
            push!(partition_local[value], index)
        end
        copy!(partition, partition_local)
    end
  
    return logLikelihood # return assignment labels and permuted indices
end

sample(
    guide::SimpleClustering, 
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