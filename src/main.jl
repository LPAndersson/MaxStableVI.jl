using MaxStableVI
using Flux

#Generate test data
coordinates = rand(10,2) #Not used for logistic model
observations = sample(
    BrownResnickModel(lambda = 0.5, nu = 0.5), 
    coordinates = coordinates, 
    n = 10
    )
data = [observations, coordinates]

resturantGuide = RestaurantProcess(delta = 0.5, 
                                   alpha = 0.5, 
                                   rho = 0.5
                                   )

guideOptimiser = Flux.Descent(1e-5)
modelOptimiser = Flux.Momentum(1e-5,0.9)

model = BrownResnickModel(lambda = 0.7, nu = 0.7)

fit = train!(model,
            resturantGuide;
            data = data,
            epochs = 10, 
            M = 8,
            guideopt = guideOptimiser,
            modelopt = modelOptimiser,
            printing = true
            );

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
