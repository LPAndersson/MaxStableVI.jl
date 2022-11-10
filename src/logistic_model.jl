import Optim
import Random

import Flux: @functor
import StatsFuns: logistic

mutable struct LogisticModel <: AbstractMaxStableModel
    theta::Vector{Float64}
end

LogisticModel(; theta::Float64) = LogisticModel([theta])

@functor LogisticModel

function clamp!(model::LogisticModel)
    model.theta[1] = clamp(model.theta[1],0.01, 0.99)
    return Nothing
end

function condLogLikelihood(
    model::LogisticModel, 
    observation::Vector{Float64}, 
    coordinates::Matrix{Float64},
    partition::Vector{Vector{Int64}}
    )
    
    d = length(observation)
    partLength = length(partition)
    tauRes = 0

    theta = model.theta[1]

    xSum = sum(observation.^(-1 / theta))
  
    for j in 1:partLength
      xTau = observation[partition[j]]
      tauLength = length(xTau)
      if tauLength == 1
        #Here is a hack to turn xTau in to scalar.
        tauRes = tauRes + ( -1 / theta - 1) * log(reshape(xTau, 1)[1]) + (theta - tauLength) * log(xSum)
      else
        r = collect(1:(tauLength-1))
        p1 = log( (-1)^(tauLength + 1) * prod(1 .- r / theta) )
        p2 = ( -1 / theta - 1) * sum(log.(xTau))
        p3 = (theta - tauLength) * log(xSum)
        tauRes = tauRes + p1 + p2 + p3
      end
    end
  
    return tauRes - xSum^theta
end

function logLikelihood( #Shi: MULTIVARIATE EXTREME VALUE DISTRIBUTION AND ITS FISHER INFORMATION MATRIX
    model::LogisticModel,
    data::Vector{Matrix{Float64}},
    ) 

    observations = data[1]

    dataDim = size(observations)[2]
    numOfObs = size(observations)[1]

    theta = model.theta[1]

    Qf = Qfcn(dataDim, theta)

    logl = 0

    for i = 1:numOfObs
        obs = observations[i,:]
        z = sum(obs.^(- 1 / theta))^theta
        Q = Qf(z)
        logl = logl + (- 1 / theta - 1) * sum(log.(obs)) + (1 - dataDim / theta) * log(z) - z + log(Q)
    end
    return logl
end

function mle!(model::LogisticModel; data::Vector{Matrix{Float64}})

    modelCopy = deepcopy(model)

    loss2 = function(x::Vector{Float64})
        modelCopy.theta = logistic.(x)
        return -logLikelihood(modelCopy, data)
    end

    optimal = Optim.optimize(loss, [0.0] , Optim.NelderMead())

    model.theta = logistic.(optimal.minimizer)
end

function Qfcn(P::Integer, alpha::Float64)
    Q = [1]
    for p in 2:P
        q1 = [((p - 1) / alpha - 1.) * Q; 0.]
        q2 = [0.; Q]
        q3 = [Q .* [0:length(Q)-1;]; 0.]
        Q = q1 + q2 - q3
    end

    retFcn = function(z)
        sum(z.^[0:(P - 1);] .* Q)
    end
end

function sample(rng::Random.AbstractRNG, model::LogisticModel; coordinates::Matrix{Float64}, n::Int64)

    d = size(coordinates)[1]
    sample = Array{Float64}(undef, n, d)

    theta = model.theta[1]

    for i in 1:n
      s = sampleStableDist(theta)
      for j in 1:d
        e = log( -log( rand(rng) ))
        sample[i, j] = exp(theta*(s-e))
      end
    end
    
    return sample

end
sample(model::LogisticModel; coordinates::Matrix{Float64}, n::Int64) = sample(Random.default_rng(), model, coordinates = coordinates, n = n)

function sampleStableDist(rng::Random.AbstractRNG, theta::Float64)

    if theta == 1
      out =  1
    else
        u = pi*rand(rng)
        w = log( -log( rand(rng) ))
        out = ((1-theta)/theta) * (log( sin((1-theta)*u) ) - w) + log( sin(theta*u) ) - 1/theta*log( sin(u) )
    end

    return out

  end
  sampleStableDist(theta::Float64) =  sampleStableDist(Random.default_rng(), theta)