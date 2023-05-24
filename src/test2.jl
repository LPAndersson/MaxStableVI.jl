using MaxStableVI
using Flux
import Optim
using Plots

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
compositeMle!(model, data = data, degree = 2)

# model = BrownResnickModel(lambda = 0.5, nu = 0.5)
# mle!(model, data = data)

loglik = function(lambda, nu)
    modelCopy = deepcopy(model)


    modelCopy.lambda = [lambda]
    modelCopy.nu = [nu]
    return compositeLogLikelihood(modelCopy, data, 2)
end

x = range(0, 2, length=100)
y = loglik.(1.54,x)
plot(x,y)