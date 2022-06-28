import QuasiMonteCarlo
import ChainRulesCore
import Distributions

function qsimvnv(Σ::Matrix{Float64},b::Vector{Float64})

	m = 100

	c = cholFactorisation(Σ)
	lb = length(b)
	unitnorm = Distributions.Normal()

	ws = QuasiMonteCarlo.sample(m, zeros(size(b)), ones(size(b)), QuasiMonteCarlo.SobolSample())

	y = Vector{Float64}(undef,lb)
	e = Vector{Float64}(undef,lb)

	int = 0.0

	for i in 1 : m

		w = ws[:,i]

		e[1] = Distributions.cdf(unitnorm,b[1]/c[1,1])

		for k in 2:lb
			y[k-1] = Distributions.quantile(unitnorm, w[k-1]*e[k-1])

			s = 0.0
			for j in 1:(k-1)
				s = s + c[k,j]*y[j]
			end

            e[k] = Distributions.cdf(unitnorm,(b[k]-s)/c[k,k])

		end

		y[lb] = Distributions.quantile(unitnorm, w[lb]*e[lb])

		int = int + prod(e)
	end

	return int/m

end

function ChainRulesCore.rrule(::typeof(qsimvnv), Σ::Matrix{Float64}, b::Vector{Float64})

	primal = qsimvnv(Σ,b)

	function qsimvnv_pullback(ȳ)

		m = 100
		c = cholFactorisation(Σ)

		lb = length(b)
		unitnorm = Distributions.Normal()

		invΣ = inv(Σ)
		invc = inv(c)
		transInvc = transpose(invc)

		ws = QuasiMonteCarlo.sample(m, zeros(size(b)), ones(size(b)), QuasiMonteCarlo.SobolSample())

		y = Vector{Float64}(undef,lb)
		e = Vector{Float64}(undef,lb)

		primal = 0.0
		gradΣ = zeros(size(Σ))
		gradb = zeros(size(b))

		for i in 1 : m

			w = ws[:,i]
			e[1] = Distributions.cdf(unitnorm,b[1]/c[1,1])

			for k in 2:lb
				y[k-1] = max( Distributions.quantile(unitnorm, w[k-1]*e[k-1]), -100)

				s = 0.0
				for j in 1:(k-1)
					s = s + c[k,j]*y[j]
				end

				e[k] = Distributions.cdf(unitnorm,(b[k]-s)/c[k,k])

			end

			y[lb] = max( Distributions.quantile(unitnorm, w[lb]*e[lb]) , -100)

			prode = prod(e)

			primal = primal + prode
			gradΣ = gradΣ .+ 0.5 * prode .*( transInvc * y * y' * invc -  invΣ)
			gradb = gradb .- prode .* transInvc * y

		end

		return ChainRulesCore.NoTangent(), ȳ*gradΣ/m, ȳ*gradb/m
	end

    return primal, qsimvnv_pullback
end

cholFactorisation = function(A::Matrix{Float64})
# Input:    Matrix A
# Output:   Cholesky factorisation of A
# LinearAlgebra.isposdef(A) || @warn "covariance matrix Σ fails positive definite check"

	L = zeros(size(A))
	n = size(A)[1]

	for j in 1:n
		sum = 0.0
		for k in 1:j
			sum = sum + L[j,k]*L[j,k]
		end

		L[j,j] = sqrt(A[j,j] - sum)

		for i in (j+1):n
			sum = 0.0
			for k in 1:j
				sum = sum + L[i,k]*L[j,k]
			end

			L[i,j] = (1.0 / L[j,j] * (A[i,j]- sum))
		end

	end

	L

end