using MaxStableVI
using Flux

#Generate test data
coordinates = rand(10,2) #Not used for logistic model
observations = sample(
    BrownResnickModel(0.5, 0.5), 
    coordinates = coordinates, 
    n = 10
    )
data = [observations, coordinates]

guide = RestaurantProcess(0.5, 0.5, 0.5)

guideOptimiser = Flux.Descent(1e-5)
modelOptimiser = Flux.Momentum(1e-5,0.9)

model = BrownResnickModel(0.7,0.7)

fit = train!(model, guide, data; 
            epochs = 1000, 
            numSamplesIn = 4,
            guideopt = guideOptimiser,
            modelopt = modelOptimiser
            )

model_mle = LogisticModel(0.8)

mle!(model_mle, data)

simulatedElbo = elboMC(model, 
guide, 
data,
10;
numSamplesIn= 4)


llenum = loglikelihoodEnumerate(model, data)

logLikelihoodIS(
    model, 
    guide, 
    data, 
    numOfSamples = 100)

using Plots
