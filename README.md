This is the [Julia](https://julialang.org/downloads/) code accompanying the paper ???

Here's a simple example to try it out:
```julia
import Pkg;
Pkg.add(url="https://github.com/LPAndersson/MaxStableVI.jl") # install this package
using MaxStableVI
using Flux

# generate some test data
coordinates = rand(10,2) # not used for logistic model
observations = sample(
    BrownResnickModel(lambda = 0.5, nu = 0.5), 
    coordinates = coordinates, 
    n = 10
    )
data = [observations, coordinates]

# initiate a restaurant process as guide
resturantGuide = RestaurantProcess(delta = 0.5, 
                                   alpha = 0.5, 
                                   rho = 0.5
                                   )

# initiate the optimizers
guideOptimiser = Flux.Descent(1e-5)
modelOptimiser = Flux.Momentum(1e-5,0.9)

# initiate the model to be trained with starting values
model = BrownResnickModel(lambda = 0.7, nu = 0.7)

# train the model
fit = train!(model,
            resturantGuide;
            data = data,
            epochs = 100, 
            M = 8,
            guideopt = guideOptimiser,
            modelopt = modelOptimiser
            );
```

For the logistic model maximum likelihood estimation is also available
```julia
model_mle = LogisticModel(theta = 0.8)
mle!(model_mle, data = data)
```