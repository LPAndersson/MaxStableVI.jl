import Random
import Zygote

import Flux: @functor

mutable struct RestaurantProcess <: AbstractGuide
    delta::Vector{Float64}
    alpha::Vector{Float64}
    rho::Vector{Float64}
end

RestaurantProcess(; delta::Float64, alpha::Float64, rho::Float64) = RestaurantProcess([delta],[alpha],[rho])

@functor RestaurantProcess

function clamp!(guide::RestaurantProcess)

    guide.delta[1] = clamp(guide.delta[1], 0.0, 0.99 )

    guide.alpha[1] = clamp(guide.alpha[1], -guide.delta[1]+0.01, Inf )
    guide.rho[1] = clamp(guide.rho[1], 0.01, Inf )

    return Nothing

end

function corrMatrixFun(guide::RestaurantProcess, observation::Vector{Float64})

    d = length(observation)
    rho = guide.rho[1]

    Sigma = Zygote.Buffer(zeros(typeof(rho), d, d))
  
    for i in 1:d
        for j in (i+1):d
            distance = abs(observation[i] - observation[j]) # compute distances
            corr = exp(- distance / rho) + 1e-100 * reshape(Random.rand(1), 1)[1] # exponential similarity  
            Sigma[i, j] = corr
            Sigma[j, i] = corr
        end
    end

    return copy(Sigma)

end

function sample(
    rng::Random.AbstractRNG,
    guide::RestaurantProcess, 
    observation::Vector{Float64},
    coordinate::Matrix{Float64},
    partition::Vector{Vector{Int64}},
    obsNum::Int64
    )
  
    delta = guide.delta[1]
    alpha = guide.alpha[1]
    rho = guide.rho[1]

    d = length(observation)

    partition_local = Vector{Vector{Int64}}(undef,0)
    reorder = Zygote.Buffer(zeros(Int64,d))

    fisher_yates_sample!(rng, 1:d, reorder) #Sample without replacement

    logLikelihood = 0.0

    Sigma = corrMatrixFun(guide, observation) # computing correlation matrix based on distances
    
    Zygote.@ignore push!(partition_local, [reorder[1]]) #put first customer at empty table

    for customerNumber in 2:d
  
        sumDistance = sum(Sigma[reorder[1:(customerNumber-1)], reorder[customerNumber]]) # sum of distances between customer and all other seated customers
        denom = alpha + customerNumber - 1 # denominator in crp formula
        probOldTable = ( (customerNumber - 1) - delta * length(partition_local) ) / denom # probability of existing table

        u = Random.rand(rng)
        cumprob = 0.0

        for table in 1:length(partition_local)
            tableDistance = 0.0
            for customerAtTable in partition_local[table] # sum of distances between value i and all values at table j
                tableDistance += Sigma[customerAtTable, reorder[customerNumber]]
            end
            tableProb = probOldTable * (tableDistance / sumDistance) # probability of being seated at table j
            cumprob += tableProb
            if u < cumprob #seat customer at existing table
                Zygote.@ignore push!(partition_local[table], reorder[customerNumber])
                logLikelihood += log(tableProb)
                break
            end
        end

        if u >= cumprob # seat customer at new table
            Zygote.@ignore push!(partition_local, [reorder[customerNumber]])
            logLikelihood += log(1-cumprob)
        end
    end

    Zygote.ignore() do 
        copy!(partition, partition_local)
    end
  
    return logLikelihood # return assignment labels and permuted indices
end

sample(
    guide::RestaurantProcess, 
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
    obsNum::Int64
    )


function fisher_yates_sample!(rng::Random.AbstractRNG, a::UnitRange{Int64}, x::Zygote.Buffer{Int64, Vector{Int64}})
    n = length(a)
    k = length(x)
    k <= n || error("length(x) should not exceed length(a)")

    inds = Zygote.Buffer(1:n)

    for i = 1:n
        @inbounds inds[i] = i
    end

    @inbounds for i = 1:k
        j = rand(rng, i:n)
        t = inds[j]
        inds[j] = inds[i]
        inds[i] = t
        x[i] = a[t]
    end
    return x
end
fisher_yates_sample!( a::UnitRange{Int64}, x::Zygote.Buffer{Int64, Vector{Int64}}) = fisher_yates_sample!(Random.default_rng(), a, x)