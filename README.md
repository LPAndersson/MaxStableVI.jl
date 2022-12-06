This is the [Julia](https://julialang.org/downloads/) code accompanying the paper [Variational inference for max-stable processes](https://arxiv.org).

Here's a simple example to try it out:
```julia
import Pkg;
Pkg.add(url="https://github.com/LPAndersson/MaxStableVI.jl") # install this package
using MaxStableVI
using Flux

# generate some test data
coordinates = rand(10,2)
observations = sample(
    #BrownResnickModel(lambda = 0.5, nu = 0.5), 
    LogisticModel(theta = 0.9),
    coordinates = coordinates, 
    n = 10
    )
data = [observations, coordinates]

# initiate a restaurant process as guide
guide = RestaurantProcess(delta = 0.9, 
                                   alpha = -0.8, 
                                   rho = 100.0
                                   )

# initiate the optimizers
guideOptimiser = Flux.AdaDelta()
modelOptimiser = Flux.AdaDelta()

# initiate the model to be trained with starting values
#model = BrownResnickModel(lambda = 0.7, nu = 0.7)
model = LogisticModel(theta = 0.7)

# train the model
fit = train!(
    model,
    guide,
    data = data,
    epochs = 5000, 
    M = 16,
    guideopt = guideOptimiser,
    modelopt = modelOptimiser
    );
```
The elbo can be estimated using Monte Carlo
```julia
elbo = elboMC(
    model, 
    guide, 
    data = data,
    numOfSamples = 100,
    M = 16
    )
```
We can also estimate the loglikehood by importance sampling the partitions from the guide distribution
```julia
logl = logLikelihoodIS(
    model, 
    guide, 
    data = data, 
    numOfSamples = 100
    )
```
For the logistic model the likelihood can be calculated exactly
```julia
logl = logLikelihood(model, data)
```
For convenience there is a function to calculate the maximum likelihood estimate
```julia
mle!(model, data = data)
```
If the dimension is low, the likelihood can also be calculated by enumerating the partitions
```julia
coordinates = rand(5,2)
observations = sample(
    BrownResnickModel(lambda = 0.5, nu = 0.5), 
    coordinates = coordinates, 
    n = 10
    )
data = [observations, coordinates]
logl = loglikelihoodEnumerate(
    BrownResnickModel(lambda = 0.5, nu = 0.5), 
    data
    )
```