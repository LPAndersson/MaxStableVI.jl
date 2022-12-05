import Random
import LinearAlgebra
import Optim

import Distributions: MvNormal
import StatsFuns: normcdf, logistic

import InvertedIndices: Not
import Flux: @functor

include("mvtGaussianCdf.jl")

mutable struct BrownResnickModel <: AbstractMaxStableModel
    lambda::Vector{Float64}
    nu::Vector{Float64}
end

BrownResnickModel(; lambda::Float64, nu::Float64) = BrownResnickModel([lambda],[nu])

@functor BrownResnickModel

function clamp!(model::BrownResnickModel)
  model.lambda[1] = clamp(model.lambda[1],0.01, Inf )
  model.nu[1] = clamp(model.nu[1], 0.01, 2.0)
  return Nothing
end

function condLogLikelihood(
    model::BrownResnickModel, 
    observation::Vector{Float64}, 
    coordinates::Matrix{Float64},
    partition::Vector{Vector{Int64}}
    )

    d = length(observation)
  
    lambda = model.lambda[1]
    nu = model.nu[1]
  
    # Distance based correlation/covariance matrix
    covDistMat = semiVarFun(lambda, nu; coordinates = coordinates, d = d)
  
    # partial derivatives of V term (from Wadsworth & Tawn (2014))
    VPartDeriv = VPartDerivBR(observation, partition, covDistMat, d)
  
    # V term
    V = VBR(observation, d, covDistMat)
  
    # Log likelihood
    return VPartDeriv - V
end

function sample(
    rng::Random.AbstractRNG, 
    model::BrownResnickModel; 
    coordinates::Matrix{Float64}, 
    n::Int
    )

    N = size(coordinates)[1]
    samples = Array{Float64}(undef, n, N)
    λ = model.lambda[1]
    ν = model.nu[1]
    γ(x) = (sqrt(sum(x.^2))/λ)^ν

    Wdist = MvNormal( covMatrix(model, coordinates) )
  
    for i in 1:n
        ζinv = -log(rand(rng))
        W = rand(rng, Wdist)
        Y = exp.( W .- W[1] - map(γ,eachrow(coordinates .- coordinates[1:1,:])) )
        Z = Y/ζinv
        for n in 2:N
            ζinv = -log(rand(rng))
            while 1/ζinv > Z[n]
                W = rand(rng, Wdist)
                Y = exp.( W .- W[n] - map(γ,eachrow(coordinates .- coordinates[n:n,:])) )
                if sum( Y[1:(n-1)].> ζinv * Z[1:(n-1)] ) == 0
                    Z = max.(Z,Y/ζinv)
                end
                E = -log(rand(rng))
                ζinv = ζinv + E
            end
        end
        samples[i,:] = Z
    end
  
    return samples
end
sample(model::BrownResnickModel; coordinates::Matrix{Float64}, n::Int) = sample(Random.default_rng(), model; coordinates = coordinates, n = n)

function semiVarFun(lambda::Float64, nu::Float64; coordinates::Matrix{Float64}, d::Int64)

    covarMat = Zygote.Buffer(zeros(typeof(lambda), d, d))
  
    for k in 1:d
        for l in 1:d
            if k == l
                covarMat[k, l] = 2 * (sqrt(sum(coordinates[k, :].^2)) / lambda)^nu
            else
                var1 = sqrt(sum(coordinates[k, :].^2))
                var2 = sqrt(sum(coordinates[l, :].^2))
                covars = sqrt(sum((coordinates[k, :] .- coordinates[l, :]).^2))
  
                covarMat[k, l] = (var1 / lambda)^nu + (var2 / lambda)^nu - (covars / lambda)^nu
            end
        end
    end
  
    return copy(covarMat)
end

function VBR(observation::Vector{Float64}, d::Int64, covDistMat::Matrix{Float64})

    V = 0.0
    # loop over all variables
    for i in 1:d
        x = observation[i] # take out variable i
        xVec = observation[Not(i)] # take out all varialbes except variable i
  
        TMat = Zygote.@ignore hcat(LinearAlgebra.I(d-1)[:, 1:(i-1)] , fill(-1, (d-1)), LinearAlgebra.I(d-1)[:, i:end]) # insert the -1 vector in column i in the identity matrix IMat
  
        covMatI = TMat * covDistMat * LinearAlgebra.transpose(TMat)
  
        varVec = LinearAlgebra.diag(covDistMat) # vector of variances for all variables except variable i
        xiVar = varVec[i]
        xjVar = varVec[Not(i)]
        covVec = covDistMat[:, i][Not(i)] # get vector of covariaces between variable i and remaining variables
  
        intUp = vec( log.(xVec ./ x) + xjVar ./ 2 .+ xiVar / 2 .- covVec )# upper integration limit
  
        gaussPr = qsimvnv(covMatI, intUp)[1] # D-1 multivariate Gaussian probability
  
        V += (1/x * gaussPr)
    end
    return V
end
  
function VPartDerivBR(
    data::Vector{Float64}, 
    partition::Vector{Vector{Int64}}, 
    covDistMat::Matrix{Float64}, 
    d::Int64
    )
  
    partLength = length(partition)

    vars = LinearAlgebra.diag(covDistMat) # variances of each variable in partition
    covDistMatInv = LinearAlgebra.inv(covDistMat)
    q = sum(covDistMatInv, dims  = 2)
    qq = q * LinearAlgebra.transpose(q)
    q1 = sum(q)
    AMat = covDistMatInv - qq ./ q1

    # partial derivatives of V term (from Wadsworth & Tawn (2014))
    VDeriv = 0
    # loop over all subsets in partition
    for j in 1:partLength
        # Computing quantities of the partial derivatives
        partInds = partition[j] # get indices for variables in partition subset j
        xTau = data[partInds] # get variables in partition subset j
        xRest = data[Not(partInds)] # get variables in partition subset j
        logXTau = log.(xTau)

        varTau = vars[partInds]

        tauLength = length(xTau) # number of variables in partition subset j
        covDistMatTau = covDistMat[partInds, partInds] # subset covariance matrix based on variables in subset j
        covDistMatTauInv = LinearAlgebra.inv(covDistMatTau)

        qTau = sum(covDistMatTauInv, dims = 2)
        qqTau = qTau * LinearAlgebra.transpose(qTau)
        qTau1 = sum(qTau)

        if length(xRest) == 0
            gaussProb = 1.0
        else
            gammaMat = LinearAlgebra.inv( AMat[ Not(partInds), Not(partInds) ] )
            gammaMat = (gammaMat + LinearAlgebra.transpose(gammaMat))/2 # Make it symmetric

            muP1 = AMat[ Not(partInds), partInds ] * logXTau
            muP2 = ( (1.0 / q1) .* q - (0.5 / q1) .* (qq * vars) + 0.5 .* (covDistMatInv * vars) )[Not(partInds),:] # non-stationary

            mu = -gammaMat * (muP1 + muP2) # mean vector for the multivariate Gaussian distribution
            intUp = vec(log.(xRest) - mu) # upper integration limit

            # if subset j only has one variabe compute univariate Gaussian, else compute multivariate Gaussian
            if length(intUp) == 1
                gaussProb = normcdf(0.0, gammaMat[1,1], intUp[1])
            else
                gaussProb = qsimvnv(gammaMat, intUp)[1]
            end
  
        end
  
        # Computing the partial derivatives term of the log-likelihood
        # each factor represents one line in the partial derivative expression on pages 8-9 in Wadsworth & Tawn (2014)
        # factor 1
        logDenom = (tauLength - 1) / 2 * log(2 * pi) + 1/2 * log(LinearAlgebra.det(covDistMatTau)) + 1/2 * log(qTau1) + sum(log.(xTau))
        fact1 = log(gaussProb + 1e-300) - logDenom
  
        if tauLength > 1
            # factor 2
            p21 = 1/4 * LinearAlgebra.transpose(varTau) * covDistMatTauInv * varTau
            p22 = 1/4 * (LinearAlgebra.transpose(varTau) * qqTau * varTau) / qTau1
            p23 = reshape((LinearAlgebra.transpose(varTau) * qTau) / qTau1 .- 1 / qTau1)[1]
            fact2 = - 1/2 * (p21 - p22 + p23)

            # factor 3
            AMatTau = covDistMatTauInv - qqTau / qTau1
            p31 = LinearAlgebra.transpose(logXTau) * AMatTau * logXTau
            p32 = reshape(LinearAlgebra.transpose(logXTau) * ((2 * qTau) / qTau1 + covDistMatTauInv * varTau - (qqTau * varTau) / qTau1))[1]
            fact3 = - 1/2 * (p31 + p32)

            # full partial derivative expression
            VDeriv += (fact1 + fact2 + fact3)
        else
            fact3 = - 1/2 * reshape(LinearAlgebra.transpose(logXTau) * ((2 * qTau) / qTau1 + covDistMatTauInv * varTau - (qqTau * varTau) / qTau1))[1]

            # full partial derivative expression
            VDeriv += (fact1 + fact3)
        end
    end
    return VDeriv
end

covMatrix = function(model::BrownResnickModel, coordinates::Matrix{Float64})

    N = size(coordinates)[1]
  
    λ = model.lambda[1]
    ν = model.nu[1]

    γ(x) = (sqrt(sum(x.^2))/λ)^ν
  
    covarMat = zeros(Float64, N, N)
  
    c = coordinates./λ
  
    for k in 1:N
      for l in 1:N
        covarMat[k, l] = γ(c[k,:]) + γ(c[l,:]) - γ(c[l,:] .- c[k,:])
      end
    end
  
    return covarMat

end

function mle!(model::BrownResnickModel; data::Vector{Matrix{Float64}})

    modelCopy = deepcopy(model)

    function loss(x)
        modelCopy.lambda = [exp(x[1])]
        modelCopy.nu = [2* logistic(x[2])]
        return -loglikelihoodEnumerate(modelCopy, data)
    end

    optimal = Optim.optimize(
        loss, 
        [0.0,0.0] , 
        Optim.NelderMead()
        )

    model.lambda = [exp(optimal.minimizer[1])]
    model.nu = [2* logistic(optimal.minimizer[2])]

    model

end