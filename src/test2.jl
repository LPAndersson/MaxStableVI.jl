using MaxStableVI
using Flux
import Optim

# generate some test data
D = 5
coordinates = rand(D,2)
observations = sample(
    BrownResnickModel(lambda = 0.5, nu = 0.5), 
    #LogisticModel(theta =  0.3),
    coordinates = coordinates, 
    n = 10
    )
data = [observations, coordinates]
#model = LogisticModel( theta = 0.5)

model = BrownResnickModel(lambda = 0.5, nu = 0.5)
compositeMle!(model, data = data, degree = 5)

#model = LogisticModel( theta = 0.5)
#mle!(model, data = data)
