This is the [Julia](https://julialang.org/downloads/) code accompanying the paper ???

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
restaurantGuide = RestaurantProcess(delta = 0.5, 
                                   alpha = 0.5, 
                                   rho = 0.5
                                   )

# initiate the optimizers
guideOptimiser = Flux.Descent(1e-6)
modelOptimiser = Flux.Momentum(1e-5,0.9)

# initiate the model to be trained with starting values
#model = BrownResnickModel(lambda = 0.7, nu = 0.7)
model = LogisticModel(theta = 0.7)

# train the model
fit = train!(
    model,
    restaurantGuide,
    data = data,
    epochs = 1000, 
    M = 8,
    guideopt = guideOptimiser,
    modelopt = modelOptimiser
    );
```
The elbo can be estimated using Monte Carlo
```julia
logl = elboMC(
    model, 
    restaurantGuide, 
    data = data,
    numOfSamples = 100,
    M = 4
    )
```
We can also estimate the loglikehood by importance sampling the partitions from the guide distribution
```julia
logl = logLikelihoodIS(
    model, 
    restaurantGuide, 
    data = data, 
    numOfSamples = 100
    )
```
For the logistic model the likelihood can be calculated exactly
```julia
observations = sample(
    LogisticModel(theta = 0.7), 
    coordinates = coordinates,
    n = 100
    )
data = [observations, coordinates]

logl = logLikelihood(LogisticModel(theta = 0.5), data)
```
For convenience there is a function to calculate the maximum likelihood estimate
```julia
model = LogisticModel(theta = 0.5)
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