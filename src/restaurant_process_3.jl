import Random
import Zygote

import Flux: @functor

mutable struct RestaurantProcess3 <: AbstractGuide
    delta::Vector{Float64}
    alpha::Vector{Float64}
    rho::Vector{Float64}
    latentEmbedding::Matrix{Float64}
end

RestaurantProcess3(; delta::Vector{Float64}, alpha::Vector{Float64}, rho::Vector{Float64},latentEmbedding::Matrix{Float64}) = RestaurantProcess3(delta,alpha,rho,latentEmbedding)

@functor RestaurantProcess3

function clamp!(guide::RestaurantProcess3)

    for i in eachindex(guide.delta)
        guide.delta[i] = clamp(guide.delta[i],0.0, 0.99)
    end
    for i in eachindex(guide.rho)
        guide.rho[i] = clamp(guide.rho[i], 0.01, Inf )
    end
    for i in eachindex(guide.alpha)
        guide.alpha[i] = clamp(guide.alpha[i], -guide.delta[i].+0.01, Inf)
    end

    return Nothing

end

function corrMatrixFun(guide::RestaurantProcess3, rho::Float64)

    embed = guide.latentEmbedding

    d = size(embed)[1]
    Sigma = Zygote.Buffer(zeros(typeof(embed[1,1]), d, d))
  
    for i in 1:d
        for j in (i+1):d

            distance = sum((embed[i,:].-embed[j,:]).^2)
            #corr = exp(- distance/rho) + 1e-100 * reshape(Random.rand(1), 1)[1] # exponential similarity
            corr = rho/(distance  + 0.001)#+ 1e-100 * reshape(Random.rand(1), 1)[1]
            
            Sigma[i, j] = corr
            Sigma[j, i] = corr
        end
    end

    return copy(Sigma)

end

function sample(
    rng::Random.AbstractRNG,
    guide::RestaurantProcess3, 
    observation::Vector{Float64},
    coordinate::Matrix{Float64},
    partition::Vector{Vector{Int64}},
    obsNum::Int64
    )
  
    delta = guide.delta[obsNum]
    alpha = guide.alpha[obsNum]
    rho = guide.rho[obsNum]

    d = length(observation)

    partition_local = Vector{Vector{Int64}}(undef,0)
    reorder = Zygote.Buffer(zeros(Int64,d))

    fisher_yates_sample!(rng, 1:d, reorder) #Sample without replacement

    logLikelihood = 0.0

    Sigma = corrMatrixFun(guide, rho) # computing correlation matrix based on distances
    
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
    guide::RestaurantProcess3, 
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