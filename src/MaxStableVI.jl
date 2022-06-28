module MaxStableVI

#Guides
abstract type AbstractGuide end

export sample, logLikelihood, RestaurantProcess
include("restaurant_process.jl")

#Models
abstract type AbstractMaxStableModel end

export condLogLikelihood, logLikelihood, sample
        
export LogisticModel, mle!
include("logistic_model.jl")

export BrownResnickModel
include("brown_resnick_model.jl")

#Model fitting
export train!
include("train.jl")

#Monte Carlo simulation
export elboMC, logLikelihoodIS, loglikelihoodEnumerate
include("montecarlo.jl")

end
