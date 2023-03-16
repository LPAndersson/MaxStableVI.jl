import StatsFuns: logistic
import Zygote
using Random
using Flux

struct NNGuide <: AbstractGuide
    W1
    b1
    W2
    b2
end
@Flux.functor NNGuide
NNGuide(in::Integer, fan::Integer) = NNGuide(randn(fan,in), randn(fan), randn(fan,fan), randn(fan))
(m::NNGuide)(x) = sum(tanh.(m.W2 * tanh.(m.W1 * x .+ m.b1 )  + m.b2))

function clamp!(guide::NNGuide)
    return Nothing
end

function sample(
    rng::Random.AbstractRNG,
    guide::NNGuide, 
    observation::Vector{Float64},
    coordinate::Matrix{Float64},
    tables::Vector{Vector{Int64}}
    )

    d = length(observation)

    tables_local = Vector{Vector{Int64}}(undef,0)
    reorder = Zygote.Buffer(zeros(Int64,d))
    rng = Random.default_rng()
    fisher_yates_sample!(rng, 1:d, reorder) #Sample without replacement

    logLikelihood = 0.0

    Zygote.@ignore push!(tables_local, [reorder[1]]) #put first customer at empty table

    for customerNumber in 2:d
        table_weights = Vector{Float64}(undef,0)
        
        for i in eachindex(tables_local)
            many_hot = Zygote.Buffer([1],d)
            for j in eachindex(many_hot)
                many_hot[j] = 0
            end
            for index in tables_local[i]
                many_hot[index] = 1.0
            end
            many_hot[customerNumber] = 1.0
            table_weights = vcat(table_weights, guide(copy(many_hot))[1])
        end

        many_hot = Zygote.Buffer([1],d)
        for j in eachindex(many_hot)
            many_hot[j] = 0
        end
        many_hot[customerNumber] = 1.0
        table_weights = vcat(table_weights, guide(copy(many_hot))[1])

        table_probs = softmax(table_weights)

        u = Random.rand(rng)
        cumprob = 0.0
        for i in eachindex(tables_local)
            cumprob += table_probs[i]
            if u < cumprob #seat customer at existing table
                Zygote.@ignore push!(tables_local[i], reorder[customerNumber])
                logLikelihood += log(table_probs[i])
                break
            end
        end

        if u >= cumprob # seat customer at new table
            Zygote.@ignore push!(tables_local, [reorder[customerNumber]])
            logLikelihood += log(1-cumprob)
        end
    end

    Zygote.ignore() do 
        copy!(tables, tables_local)
    end

    return logLikelihood
end
sample(
    guide::NNGuide, 
    observation::Vector{Float64},
    coordinate::Matrix{Float64},
    partition::Vector{Vector{Int64}}
    ) = sample(
    Random.default_rng(),
    guide, 
    observation,
    coordinate,
    partition
    )