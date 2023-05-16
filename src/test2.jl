using MaxStableVI
using Flux
import Optim

# generate some test data
coordinates = rand(10,2)
observations = sample(
    BrownResnickModel(lambda = 0.5, nu = 0.5), 
    #LogisticModel(theta =  0.5),
    coordinates = coordinates, 
    n = 10
    )
data = [observations, coordinates]
#model = LogisticModel( theta = 0.5)

model = BrownResnickModel(lambda = 0.5, nu = 0.5)
compositeMle!(model, data = data, degree = 2)

model = BrownResnickModel(lambda = 0.5, nu = 0.5)
mle!(model, data = data)
