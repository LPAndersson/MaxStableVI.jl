import Zygote
import Random
import StatsBase

using StatsFuns: logsumexp
using Statistics: mean

rng =  Random.default_rng()
seed = 1235
Random.seed!(seed);

# generate some test data
coordinates = rand(5,2)
observations = sample(
    #BrownResnickModel(lambda = 0.5, nu = 0.5), 
    LogisticModel(theta = 0.9),
    coordinates = coordinates, 
    n = 5
    )
data = [observations, coordinates]

# initiate a restaurant process as guide
guide = RestaurantProcess(delta = 0.5, 
                                alpha = 0.51, 
                                rho = 0.52
                                )

model = LogisticModel(theta = 0.9)

gradEstimate1 = function(rng, model, guide, data)

    observations = data[1]
    coordinates = data[2]
    
    (n, d) = size(observations)
    
    M=5

    guideSamples = [[[1]] for _ in 1:M]

    guideValues = Vector{Float64}(undef, M)
    guideGrads = Vector{Zygote.Grads}(undef, M)

    pqgradLogq = Array{Zygote.Grads}(undef, M)
    pqgradLogp = Array{Zygote.Grads}(undef, M)

    modelValues = Vector{Float64}(undef, M)
    modelGrads = Vector{Zygote.Grads}(undef, M)
        
    guideParams = Flux.params(guide)
    modelParams = Flux.params(model)

    batchOrder = StatsBase.sample(rng, 1:n, n, replace = false)

    guideSteps = Vector{Any}(undef, n)

    epoch = 1

    for obsIdx in batchOrder
        for m in 1:M
            
            (guideValues[m], guideGrads[m]) = 
                Zygote.withgradient( 
                    () -> sample(
                            guide, 
                            observations[obsIdx,:],
                            coordinates,
                            guideSamples[m]
                            ), 
                    guideParams
                    )

            (modelValues[m], modelGrads[m]) = 
                Zygote.withgradient( 
                    () -> condLogLikelihood(
                        model, 
                        observations[obsIdx,:], 
                        coordinates, 
                        guideSamples[m]
                        ), 
                    modelParams
                    )

            pqgradLogq[m] = exp(modelValues[m] - guideValues[m]) .* guideGrads[m]
            pqgradLogp[m] = exp(modelValues[m] - guideValues[m]) .* modelGrads[m]

        end

        guideGradSum = reduce(.+, guideGrads)
        log_pqsum = logsumexp(modelValues .- guideValues)              
        guideSteps[obsIdx] =  -exp(-log_pqsum) .* reduce(.+, pqgradLogq) .+ log_pqsum .* guideGradSum
    
    end

    reduce(.+, guideSteps)

end

numSim = 10000;
grads = Vector{Any}(undef,numSim);

for count in 1:numSim

    grads[count] = gradEstimate1(rng, model, guide, data)

end

gradTot = reduce(.+, grads)./numSim;
gradTot.grads

delta = 0.1

guidePlus = RestaurantProcess(
    delta = 0.5 , 
    alpha = 0.51, 
    rho = 0.52 + delta/2.0
    )

guideMinus = RestaurantProcess(
    delta = 0.5 , 
    alpha = 0.51, 
    rho = 0.52- delta/2.0 
)

elboMinus = elboMC(
    LogisticModel(theta = 0.90), 
    guideMinus, 
    data = data,
    numOfSamples = 10000000,
    M = 5
    )

elboPlus = elboMC(
        LogisticModel(theta = 0.90), 
        guidePlus, 
        data = data,
        numOfSamples = 10000000,
        M = 5
        )

gradEst = (elboPlus - elboMinus) / delta


params = range(0.1, 0.9, length = 20)
elbos = Vector{Float64}(undef, 0)
for param in params
    append!(
        elbos, 
        elboMC(
            LogisticModel(theta = 0.90), 
            RestaurantProcess(delta = 0.5, 
                                    alpha = 0.5, 
                                    rho = param
                                    ), 
            data = data,
            numOfSamples = 100000,
            M = 1
        )
    )
end

plot(params, elbos)
