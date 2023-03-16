using MaxStableVI
using Flux

# generate some test data
coordinates = rand(10,2)
observations = sample(
    #BrownResnickModel(lambda = 0.5, nu = 0.5), 
    LogisticModel(theta =  0.5),
    coordinates = coordinates, 
    n = 10
    )
data = [observations, coordinates]

# initiate a restaurant process as guide
 #guide = NNGuide(10,50)
guide = RestaurantProcess(delta = 0.9, 
                                   alpha = -0.8, 
                                   rho = 100.0
                                   )
# initiate the optimizers
guideOptimiser = Flux.Optimise.AdaGrad(0.1, 1.0e-8)
modelOptimiser = Flux.Optimise.AdaGrad(0.1, 1.0e-8)

# initiate the model to be trained with starting values
#model = BrownResnickModel(lambda = 0.7, nu = 0.7)
model = LogisticModel( theta = 0.5)

# train the model
fit = train!(
    model,
    guide,
    data = data,
    epochs = 10000, 
    M = 1,
    guideopt = guideOptimiser,
    modelopt = modelOptimiser
    );


elbo = elboMC(
model, 
guide, 
data = data,
numOfSamples = 100,
M = 1
)

logl = logLikelihood(model, data)

mle!(model, data = data)