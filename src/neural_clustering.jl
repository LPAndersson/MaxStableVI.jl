import Random
import Flux
import Zygote
import StatsFuns
import Statistics
import Distributions
import SliceMap

import Flux: @functor

#= mutable struct NeuralClustering <: AbstractGuide
    #customers are in rows and tables in columns
    h::Vector{Float32}
    weights::Vector{Float32}
    g::Flux.Chain
end

NeuralClustering(; h::Vector{Float32},  weights::Vector{Float32}, g::Flux.Chain) = NeuralClustering(h,weights,g)
 =#
 mutable struct NeuralClustering <: AbstractGuide
    #customers are in rows and tables in columns
    h::Matrix{Float32}
    u::Matrix{Float32}
    g::Flux.Chain
    f::Flux.Chain
end

NeuralClustering(;D::Integer, dh::Integer, dg::Integer) = 
NeuralClustering(
    rand(Float32,(D,dh)).-convert(Float32,0.5),
    rand(Float32,(D,dh)).-convert(Float32,0.5),
    Flux.Chain(Flux.Dense(dh,5,Flux.relu), Flux.Dense(5,dg)),
    Flux.Chain(Flux.Dense(dg+dh,5,Flux.relu), Flux.Dense(5,1))
)

@functor NeuralClustering

function clamp!(guide::NeuralClustering)

    return Nothing

end

function sample(
    rng::Random.AbstractRNG,
    guide::NeuralClustering, 
    observation::Vector{Float64},
    coordinate::Matrix{Float64},
    partition::Vector{Vector{Int64}},
    obsNum::Int64
    )
  
    h = guide.h
    u = guide.u
    g = guide.g
    f = guide.f

    N, dh = size(h)

    c = Vector{Int32}(undef,N)
    
    logLikelihood = 0.0

    U = vec(sum(u[2:end,:],dims = 1))
    #Zygote.@ignore H[1,:] = h[1,:]
    H = [h[1:1,:];zeros(Float32,N-1,dh)]

    G = g(H[1,:])
    K = 1
    Zygote.@ignore c[1] = 1

    for n in 2:N
        U = U - u[n,:]

        log_q = SliceMap.slicemap((x) -> f([G + g(x + h[n,:]) - g(x);U]), H[1:(K+1),:], dims = 2)
        q = Flux.softmax(vec(log_q))
        Zygote.@ignore c[n] = Random.rand(rng, Distributions.Categorical(q))
        logLikelihood += log(copy(q[c[n]]))
        if c[n] == K+1
            K = K + 1
        end
        G = G .- g(H[c[n],:]) .+ g(H[c[n],:]+h[n,:])
        H = H .+ [zeros(Float32,c[n]-1,dh);h[n:n,:];zeros(Float32,N-c[n],dh)]
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
    guide::NeuralClustering, 
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
    guide::NeuralClustering, 
    observation::Vector{Float64},
    coordinate::Matrix{Float64},
    partition::Vector{Vector{Int64}},
    obsNum::Int64
    )
    
    h = guide.h
    u = guide.u
    g = guide.g
    f = guide.f

    N, dh = size(h)

    c = Vector{Int32}(undef,N)

    for (index,value) in enumerate(partition)
        for cust in value
            c[cust] = index
        end
    end
    
    logLikelihood = 0.0

    U = vec(sum(u[2:end,:],dims = 1))
    H = [h[1:1,:];zeros(Float32,N-1,dh)]

    G = g(H[1,:])
    K = 1
    Zygote.@ignore c[1] = 1

    for n in 2:N
        U = U - u[n,:]

        log_q = SliceMap.slicemap((x) -> f([G + g(x + h[n,:]) - g(x);U]), H[1:(K+1),:], dims = 2)
        q = Flux.softmax(vec(log_q))
        logLikelihood += log(copy(q[c[n]]))
        if c[n] == K+1
            K = K + 1
        end
        G = G .- g(H[c[n],:]) .+ g(H[c[n],:]+h[n,:])
        H = H .+ [zeros(Float32,c[n]-1,dh);h[n:n,:];zeros(Float32,N-c[n],dh)]
    end
    
    return logLikelihood
end