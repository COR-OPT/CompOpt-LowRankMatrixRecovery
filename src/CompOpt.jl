"""
A module implementing nonsmooth optimization methods for a variety of
statistical problems that exhibit sharpness and weak convexity.
"""
module CompOpt

    include("Utils.jl")

    using LinearAlgebra
    using Random
    using Statistics
	using SparseArrays

	# quadratic sensing problem
    struct QuadProb
        y :: Array{Float64, 1}
        X :: Array{Float64, 2}
        A :: Array{Float64, 2}
        pfail :: Float64
    end


	# symmetrized quadratic sensing problem
	struct SymQuadProb
		y :: Array{Float64, 1}
        X :: Array{Float64, 2}
        A1 :: Array{Float64, 2}
		A2 :: Array{Float64, 2}
        pfail :: Float64
	end


	# bilinear sensing problem
    struct BilinProb
        y :: Array{Float64, 1}
		W :: Array{Float64, 2}  # signal 1
        X :: Array{Float64, 2}  # signal 2
        A :: Array{Float64, 2}
        B :: Array{Float64, 2}
        pfail :: Float64
    end


	# non-euclidean robust pca formulation
	struct RpcaProb
		W :: Array{Float64, 2}  # X_{\sharp} X_{\sharp}^T + S
		X :: Array{Float64, 2}  # X_{\sharp}
		S :: Array{Float64, 2}  # S - sparse corruption matrix
		gamma :: Float64  		# radius of 2,\infty norm
		pfail :: Float64
	end


	# matrix completion
	struct MatCompProb
		X :: Array{Float64, 2}
		M :: Array{Float64, 2}
		mask :: Array{Number, 2}
		p :: Float64
	end


	#= squared norm shortcut =#
	sqnorm(x) = norm(x)^2


	"""
		genSymQuadProb(d, m, r, noise_lvl=0.0)

	Generates a symmetrized quadratic sensing problem in dimensions
	``d \\times n`` where ``\\rank(X) = r`` with a desired noise level.
	"""
	function genSymQuadProb(d, m, r, noise_lvl=0.0)
		A1 = randn(m, d); A2 = randn(m, d); X = randn(d, r)
		y = vec(mapslices(sqnorm, A1 * X, dims=[2]))
		y = y - vec(mapslices(sqnorm, A2 * X, dims=[2]))
		Utils.corrupt_measurements!(y, noise_lvl, :gaussian)
		return SymQuadProb(y, X, A1, A2, noise_lvl)
	end


    """
        genQuadProb(d, n, m, r, noise_lvl=0.0)

    Generates a quadratic sensing problem in dimensions ``d \\times n`` where
    ``\\rank(X) = r`` with a desired noise level.
    """
    function genQuadProb(d, m, r, noise_lvl=0.0)
		A = randn(m, d); X = randn(d, r)
		y = mapslices(sqnorm, A * X, dims=[2])[:]  # get measurements
        Utils.corrupt_measurements!(y, noise_lvl, :gaussian)
        return QuadProb(y, X, A, noise_lvl)
    end


    """
        genBilinProb(d1, d2, m, r, noise_lvl=0.0)

    Generates a bilinear sensing problem in dimensions ``d_1 \\times d_2``
    where ``\\rank(X) = r`` with a desired noise level.
    """
    function genBilinProb(d1, d2, m, r, noise_lvl=0.0)
		A = randn(m, d1); B = randn(m, d2); W = randn(d1, r); X = randn(d2, r)
		y = Utils.rowwise_prod(A * W, B * X)  # row-wise dot product
        Utils.corrupt_measurements!(y, noise_lvl, :gaussian)
        return BilinProb(y, W, X, A, B, noise_lvl)
    end


	"""
		genRpcaProb(d, r, corr_lvl=0.0)

	Generates a robust pca problem in dimensions ``d \\times r`` where
	a `corr_lvl` fraction of entries are outliers.
	"""
	function genRpcaProb(d, r, corr_lvl=0.0)
		X = Utils.genIncoherentMatrix(d, r)
		S = Utils.genSparseMatrix(d, d, corr_lvl)
		W = X * X' + S
		return RpcaProb(W, X, S, 2 * Utils.abNorm(X, Inf, 2), corr_lvl)
	end


	"""
		genMatCompProb(d, r, sample_freq=1.0)

	Generate a matrix completion problem in dimensions ``d \\times r`` where
	elements are sampled with `sample_freq` frequency.
	"""
	function genMatCompProb(d, r, sample_freq=1.0)
		X = Utils.genIncoherentMatrix(d, r)
		# equalize singular values according to paper
		svdObj = svd(X); S = svdObj.S; S[:] = ones(length(S)) * median(S)
		X = svdObj.U * (S .* svdObj.Vt)  # reform X
		mask = Utils.genMask(d, sample_freq)
		return MatCompProb(X, X * X', mask, sample_freq)
	end


    """
        quadRes(qProb, Xcurr)

    Compute the residual of the quadratic sensing model at the current
    estimate `Xcurr`.
    """
    function quadRes(qProb, Xcurr)
		r = mapslices(sqnorm, qProb.A * Xcurr, dims=[2])[:]
        broadcast!(-, r, r, qProb.y)
        return r
    end


	"""
		symQuadRes(qProb, Xcurr)

	Compute the residual of the symmetrized quadratic sensing model at the
	current estimate `Xcurr`.
	"""
	function symQuadRes(qProb, Xcurr)
		r = vec(mapslices(sqnorm, qProb.A1 * Xcurr, dims=[2]))
		r = r - vec(mapslices(sqnorm, qProb.A2 * Xcurr, dims=[2]))
		broadcast!(-, r, r, qProb.y)
		return r
	end


	"""
		symQuadRobustLoss(qProb, Xcurr)

	Compute the robust loss for the symmetrized quadratic sensing problem given
	a problem instance `qProb` and the current estimate `Xcurr`.
	"""
	function symQuadRobustLoss(qProb, Xcurr)
		r = symQuadRes(qProb, Xcurr)
		return (1 / length(qProb.y)) * norm(r, 1)
	end


    """
        quadRobustLoss(qProb, Xcurr)

    Compute the robust loss for the quadratic sensing problem given a problem
    instance `qProb` and the current estimate `Xcurr`.
    """
    function quadRobustLoss(qProb, Xcurr)
        r = quadRes(qProb, Xcurr)
        return (1 / length(qProb.y)) * norm(r, 1)
    end


    """
        bilinRes(bProb, Ucurr, Vcurr)

    Compute the residual of the bilinear sensing model at the current estimate
    `(Ucurr, Vcurr)`.
    """
    function bilinRes(bProb, Ucurr, Vcurr)
        r = Utils.rowwise_prod(bProb.A * Ucurr, bProb.B * Vcurr)
		broadcast!(-, r, r, bProb.y)
        return r
    end


    """
        bilinRobustLoss(bProb, Ucurr, Vcurr)

    Compute the robust loss for the bilinear sensing problem given a problem
    instance `bProb` and the current estimate `(Ucurr, Vcurr)`.
    """
    function bilinRobustLoss(bProb, Ucurr, Vcurr)
        r = bilinRes(bProb, Ucurr, Vcurr)
        return (1 / length(bProb.y)) * norm(r, 1)
    end


	"""
		matCompLoss(prob::MatCompProb, Xcurr)

	Returns the Frobenius error of the PSD matrix completion iterate.
	"""
	function matCompLoss(prob::MatCompProb, Xcurr)
		return norm(prob.mask .* (Xcurr * Xcurr' - prob.M))
	end


    """
        quadSubgrad(qProb, Xcurr)

    Compute the subgradient at `Xcurr` for the quadratic sensing problem.
    """
    function quadSubgrad(qProb, Xcurr)
		m = length(qProb.y)
        # sign and A * X
        rSign = map.(sign, quadRes(qProb, Xcurr)); R = qProb.A * Xcurr
        # compute subgradient
        return (2 / m) * qProb.A' * (rSign .* R)
    end


	"""
        symQuadSubgrad(qProb, Xcurr)

    Compute the subgradient at `Xcurr` for the symmetrized quadratic sensing
	problem.
    """
	function symQuadSubgrad(qProb, Xcurr)
		m = length(qProb.y)
		rSign = map.(sign, symQuadRes(qProb, Xcurr))
		R1 = qProb.A1 * Xcurr; R2 = qProb.A2 * Xcurr
		return (2 / m) * (qProb.A1' * (rSign .* R1) - qProb.A2' * (rSign .* R2))
	end


	"""
		symQuadGrad(qProb, Xcurr)

	Compute the gradient at `Xcurr` for the symmetrized quadratic sensing
	problem.
	"""
	function symQuadGrad(qProb, Xcurr)
		m = length(qProb.y)
		res = symQuadRes(qProb, Xcurr)
		R1 = qProb.A1 * Xcurr; R2 = qProb.A2 * Xcurr
		return (2 / m) * (qProb.A1' * (res .* R1) - qProb.A2' * (res .* R2))
	end


	"""
		symQuadNaiveGD(qProb, Xcurr, iters, sSize; eps=1e-12)

	Solve a symmetrized quadratic sensing problem using "naive" gradient
	descent, given initial estimate `Xcurr` and a step size schedule `sSize`,
	which can be either a number or a callable accepting the iteration number
	as its argument.
	"""
	function symQuadNaiveGD(qProb, Xcurr, iters, sStep; eps=1e-12)
		# note: sStep is either a callable or a Number
		stepSize = Utils.setupStep(sStep)
		dist = fill(0.0, iters); Xtrue = qProb.X
		for k = 1:iters
			dist[k] = Utils.norm_mat_dist(Xcurr * Xcurr', Xtrue * Xtrue')
			broadcast!(-, Xcurr, Xcurr, stepSize(k) * symQuadGrad(qProb, Xcurr))
			if dist[k] <= eps
				return Xcurr, dist[1:k]
			end
		end
		return Xcurr, dist
	end


	"""
		symQuadNaiveGD_init(qProb, delta, iters, sSize; eps=1e-12)

	Solve a symmetrized quadratic sensing problem using "naive" gradient
	descent, forming its initial estimate by taking a small random direction
	away from the ground truth, given a step size schedule `sSize` which can be
	either a number or a callable accepting the iteration number as its argument.
	"""
	function symQuadNaiveGD_init(qProb, delta, iters, sStep; eps=1e-12)
		Xtrue = qProb.X; d, r = size(Xtrue)
        randDir = randn(d, r); randDir = randDir / norm(randDir)
        Xinit = Xtrue + delta * randDir * norm(Xtrue)
		return symQuadNaiveGD(qProb, Xinit, iters, sStep, eps=eps)
	end


    """
        bilinSubgrad(bProb, Ucurr, Vcurr)

    Compute the subgradient at `(Ucurr, Vcurr)` for the bilinear sensing
	problem.
    """
    function bilinSubgrad(bProb, Ucurr, Vcurr)
		m = length(bProb.y)
        rSign = map.(sign, bilinRes(bProb, Ucurr, Vcurr))
		R1 = bProb.A * Ucurr; R2 = bProb.B * Vcurr;
		return (1 / m) * (rSign .* bProb.A)' * R2, (1 / m) * (rSign .* bProb.B)' * R1
    end


	"""
		bilinGrad(bProb, Ucurr, Vcurr)

	Compute the naive gradient at `(Ucurr, Vcurr)` for the bilinear sensing
	problem.
	"""
	function bilinGrad(bProb, Ucurr, Vcurr)
		m = length(bProb.y)
		res = bilinRes(bProb, Ucurr, Vcurr)
		R1 = bProb.A * Ucurr; R2 = bProb.B * Vcurr
		return (2 / m) * (res .* bProb.A)' * R2, (2 / m) * (res .* bProb.B)' * R1
	end


	"""
		bilinNaiveGD(bProb, Ucurr, Vcurr, iters, sSize; eps=1e-12)

	Solve a bilinear sensing problem using "naive" gradient descent, given
	initial estimates `Ucurr, Vcurr` and a step size schedule `sSize`, which
	can be either a number or a callable accepting the iteration number as its
	argument.
	"""
	function bilinNaiveGD(bProb, Ucurr, Vcurr, iters, sSize; eps=1e-12)
		stepSize = Utils.setupStep(sSize)
		d1, r = size(Ucurr); d2, r = size(Vcurr)
		Mtrue = bProb.W * bProb.X'
		gradU = fill(0.0, (d1, r)); gradV = fill(0.0, (d2, r))
        dist = fill(0.0, iters)
        for i = 1:iters
            dist[i] = Utils.norm_mat_dist(Ucurr * Vcurr', Mtrue)
			if dist[i] <= eps
				return Ucurr, Vcurr, dist[1:i]
			end
            gradU[:], gradV[:] = bilinGrad(bProb, Ucurr, Vcurr)
			broadcast!(-, Ucurr, Ucurr, stepSize(i) * gradU)
			broadcast!(-, Vcurr, Vcurr, stepSize(i) * gradV)
        end
        return Ucurr, Vcurr, dist
	end

	"""
		bilinNaiveGD_init(bProb, delta, iters, sSize; eps=1e-12)

	Solve a bilinear sensing problem using "naive" gradient descent, forming
	initial estimates by taking a small random direction away from the ground
	truth, given a step size schedule `sSize` which can be either a number or
	a callable accepting the iteration number as its argument.
	"""
	function bilinNaiveGD_init(bProb, delta, iters, sSize; eps=1e-12)
		Wtrue = bProb.W; Xtrue = bProb.X
		d1, r = size(Wtrue); d2, r = size(Xtrue)
		randW = randn(d1, r); randW /= norm(randW)
		randX = randn(d2, r); randX /= norm(randX)
		Uinit = Wtrue + delta * randW * norm(Wtrue)
		Vinit = Xtrue + delta * randX * norm(Xtrue)
        return bilinNaiveGD(bProb, Uinit, Vinit, iters, sSize, eps=eps)
	end
# --------------------------------------
# Matrix completion
# --------------------------------------

	"""
		matCompSubgrad(prob::MatCompProb, Xcurr)

	Compute a subgradient of the PSD matrix completion problem at the current
	estimate `Xcurr`.
	"""
	function matCompSubgrad(prob::MatCompProb, Xcurr)
		P = prob.mask
		T0 = (Xcurr * Xcurr' - prob.M); T1 = P .* T0; T2 = P' .* T0
		if all(T0 .== 0)
			return T0
		else
			return (T1 * Xcurr / norm(T2)) + (T1' * Xcurr / norm(T1))
		end
	end


	"""
		matCompGrad(prob::MatCompProb, Xcurr)

	Compute the "naive" gradient of the PSD matrix completion problem at the
	current estimate `Xcurr`.
	"""
	function matCompGrad(prob::MatCompProb, Xcurr)
		P = prob.mask
		T0 = (Xcurr * Xcurr' - prob.M); T1 = P .* T0
		return  2 * (T1 * Xcurr + T1' * Xcurr)
	end


	"""
		matCompNaiveGD(prob::MatCompProb, Xcurr, iters, sSize; eps=1e-12)

	Solve a PSD matrix completion problem using "naive" gradient descent on a
	squared Frobenius formulation, given a step size `sSize` which is either a
	number or a callable accepting the iteration number as its argument.
	"""
	function matCompNaiveGD(prob::MatCompProb, Xcurr, iters, sSize; eps=1e-12)
		stepSize = Utils.setupStep(sSize)
		dist = fill(0.0, iters)
		for i = 1:iters
			dist[i] = Utils.norm_mat_dist(Xcurr * Xcurr', prob.M)
			if dist[i] <= eps
				return Xcurr, dist[1:i]
			end
			broadcast!(-, Xcurr, Xcurr, stepSize(i) * matCompGrad(prob, Xcurr))
		end
		return Xcurr, dist
	end
	function matCompInternalSolver(prob::MatCompProb, W, Xcurr, maxIter, ϵ, ρ;eps = 1e-12)
		p = prob.p;
		t = 1;
		a = t*sqrt(p*(1+ϵ))
		b = t*sqrt(p*ϵ)
		d, r = size(Xcurr)
		m = nnz(sparse(prob.mask))
		λ = ones(d*r,1)
		ν = ones(m,1)
		w = zeros(m,1)
		x = reshape(Xcurr,d*r,1)
		z = -x
		Diff = prob.M - Xcurr*Xcurr'
		y = Diff[prob.mask .> 0]
		XkI = kron(Xcurr, zeros(d,d)+UniformScaling(1))
		A = W*XkI

		# L = cholesky(zeros(d*r,d*r)+UniformScaling(1) + A'*A)
		Q = zeros(d*r,d*r)+UniformScaling(1) + A'*A
		for ii = 1:maxIter
			nz = norm(z - λ)
			if nz < 1e-10
				zp1 = zeros(d*r,1)
			else
				zp1 = max(ρ*nz-a,0)*(z-λ)/((2*b+ρ)*nz)
			end
			nw = norm(w-ν-y)
			if nw < 1e-10
				wp1 = zeros(m,1)
			else
				wp1 = max(nw-1/ρ,0)*(w-ν-y)/nw + y;
			end
			# zp = L\(L'\(zp1+λ+A'*(wp1+ν)))
			zp = Q\(zp1+λ+A'*(wp1+ν))
			wp = A*zp

			λ = λ + zp1 - zp
			ν = ν + wp1 - wp
			res = norm(zp - z) + norm(wp - w)
			z = zp
			w = wp
			if res < eps
				break
			end
		end
		return reshape(z+x,d,r)
	end
	function matCompProxLinear(prob::MatCompProb, Xcurr, maxIter, ϵ; eps = 1e-12)
		dist = fill(0.0, maxIter)
		d, r= size(Xcurr)
		# mask = map(x->min(1,max(0,x)), prob:mask + prob:mask')
		idx = findall(x->(x.>0),prob.mask)
		m = nnz(sparse(prob.mask))
		k1 = zeros(Int64, m,1)
		k2 = zeros(Int64, m,1)
		# ϵ = 0.5

		for ii = 1:m
			k1[ii] = LinearIndices(prob.mask)[idx[ii]]
			k2[ii] = LinearIndices(prob.mask)[CartesianIndex(idx[ii][2], idx[ii][1])]
		end
		# indi = LinearIndices(prob.mask)[idx]
		W = sparse([Array(1:m); Array(1:m)], vec([k1; k2]), ones(2*m), m, d^2);
		for ii = 1:maxIter
			dist[ii] = Utils.norm_mat_dist(Xcurr*Xcurr', prob.M)
			if dist[ii] <= eps
			end
			# Xcurr = matCompInternalSolver(prob, W, Xcurr, maxIter, ϵ, log(1/dist[ii]))
			Xcurr = matCompInternalSolver(prob, W, Xcurr, maxIter, ϵ, 1/m)
		end
		return Xcurr, dist
	end


# --------------------------------------
# RPCA
# --------------------------------------

	#= Helper function for the cost of the prox-linear objective for RPCA =#
	proxlin_rpca_cost(X, Xk, C, gamma) = begin
		fcA = sum(abs.(X * Xk' + Xk * X' - C))
		fcB = (1 / (2 * gamma)) * Utils.abNorm(X - Xk, 1, 2)
		return fcA + fcB
	end


	"""
		rpcaStep(prob::RpcaProb, Xk, C; gamma=10.0, eta=0.01, maxIt=5000)

	Compute an (approximate) prox-linear step of the non-euclidean robust PCA
	formulation, given a current iterate `Xk`, an upper bound `C` on the
	``\\ell_{2 \\to \\infty}`` norm of the estimates, and optional parameters:
	- ``gamma``: the quadratic penalty parameter
	- ``eta``: the step size
	- `maxIt`: maximum number of iterations

	Uses the subgradient method internally.
	"""
	function rpcaStep(prob::RpcaProb, Xk, C; gamma=10.0, eta=0.01, maxIt=5000)
		Yk = copy(Xk); Ybest = copy(Xk); minF = Inf
		factC = Xk * Xk' + prob.W  # constant factor in 1-norm
		fCost = (X -> proxlin_rpca_cost(X, Xk, factC, gamma))
		Gquad = fill(0.0, size(Xk))  # (sub)gradient 1
		Gnorm = fill(0.0, size(Xk))  # (sub)gradient 2
		Gtot = fill(0.0, size(Xk))  # total
		for i = 1:maxIt
			# get quadratic penalty gradient
			Gquad[:] = Utils.subg_sq21Norm(Yk, Xk)
			rmul!(Gquad, 1 / gamma)
			# Yk[:] = Yk[:] - (1 / gamma) * Gquad
			# get subgradient of ell-1 elementwise norm
			Gnorm[:] = 2 * sign.(Xk * Yk' + Yk * Xk' - factC) * Xk
			Gtot[:] = Gquad + Gnorm;
			# apply subgradient step
			broadcast!(-, Yk, Yk, eta * Gtot / (norm(Gtot)^2))
			# apply projection
			Yk[:] = Utils.matProj_2inf(Yk, C)  # project back to C
			if fCost(Yk) < minF
				copyto!(Ybest, Yk); minF = fCost(Yk)  # update stuff
			end
		end
		return Ybest
	end


	"""
		rpcaProxLin(prob::RpcaProb, Xk, C, iters; gamma=10.0, eta=0.1, maxIt=5000)

	Solve a robust PCA instance using the prox-linear method for `iters`
	iterations, given an upper bound `C` on the ``2,\\infty`` norm of the
	solution. The parameter ``\\gamma`` is the scale of the norm penalty for
	the proximal subproblems, ``\\eta`` is the step size of the subgradient
	method used internally to solve each subproblem, while `maxIt` controls the
	maximum number of iterations per subproblem.
	"""
	function rpcaProxLin(prob::RpcaProb, Xk, C, iters;
						 eps=1e-12, gamma=10.0, eta=0.1, maxIt=5000)
		Yk = copy(Xk); dists = fill(0.0, iters); M = prob.X * prob.X'
		step = Utils.setupStep(eta)
		for i = 1:iters
			dists[i] = norm(Yk * Yk' - M) / norm(M)
			if dists[i] <= eps
				return Yk, dists[1:i]
			end
			# perform a prox-step
			Yk[:] = rpcaStep(prob, Yk, C, gamma=gamma, eta=step(i), maxIt=maxIt)
		end
		return Yk, dists
	end


	function rpcaProxLin_init(prob::RpcaProb, delta, iters;
							  eps=1e-12, gamma=10.0, eta=0.01, maxIt=1000)
		Xtrue = prob.X; d, r = size(Xtrue)
		randDir = randn(d, r); randDir = randDir / norm(randDir)
		Xinit = Xtrue + delta * randDir * norm(Xtrue)
		Xnorm = Utils.abNorm(Xtrue, Inf, 2)
		return rpcaProxLin(prob, Xinit, 2 * Xnorm, iters,
						   eps=eps, gamma=gamma, eta=eta, maxIt=maxIt)
	end

    """
        pSgd(qProb::QuadProb, Xinit, iters; λ = 1.0, rho = 0.98)

    Apply the projected subgradient method with geometrically decaying step
    to the quadratic sensing problem for `iters` iterations, given a problem
    instance `qProb` and an initial estimate `Xinit`.
    """
    function pSgd(qProb::QuadProb, Xinit, iters; λ = 1.0, rho = 0.98)
        Xtrue = qProb.X; d, r = size(Xtrue); grad = fill(0.0, (d, r))
        q = λ; dist = fill(0.0, iters);
        for i = 1:iters
            dist[i] = Utils.norm_mat_dist(Xinit * Xinit', Xtrue * Xtrue')
            grad[:] = quadSubgrad(qProb, Xinit)
			broadcast!(-, Xinit, Xinit, q * grad / norm(grad))
            q = q * rho
        end
        return Xinit, dist
    end


	"""
        pSgd(qProb::SymQuadProb, Xinit, iters; λ = 1.0, rho = 0.98)

    Apply the projected subgradient method with geometrically decaying step
    to the quadratic sensing problem for `iters` iterations, given a problem
    instance `qProb` and an initial estimate `Xinit`.
    """
    function pSgd(qProb::SymQuadProb, Xinit, iters;
				  λ=1.0, rho=0.98, eps=1e-10)
        Xtrue = qProb.X; d, r = size(Xtrue); grad = fill(0.0, (d, r))
        q = λ; dist = fill(0.0, iters);
        for i = 1:iters
            dist[i] = Utils.norm_mat_dist(Xinit * Xinit', Xtrue * Xtrue')
			if dist[i] <= eps
				return Xinit, dist[1:i]
			end
            grad[:] = symQuadSubgrad(qProb, Xinit)
			broadcast!(-, Xinit, Xinit, q * grad / norm(grad))
            q = q * rho
        end
        return Xinit, dist
    end


    """
        pSgd(bProb::BilinProb, Uinit, Vinit, iters; λ=1.0, rho=0.98)

    Apply the projected subgradient method with geometrically decaying step
    to the bilinear sensing problem for `iters` iterations, given a problem
    instance `qProb` and an initial estimate `(Uinit, Vinit)`.
    """
    function pSgd(bProb::BilinProb, Uinit, Vinit, iters;
				  λ=1.0, rho=0.98, eps=1e-10)
        d1, r = size(Uinit); d2, r = size(Vinit)
		Mtrue = bProb.W * bProb.X'
		gradU = fill(0.0, (d1, r)); gradV = fill(0.0, (d2, r))
        q = λ; dist = fill(0.0, iters);
        for i = 1:iters
            dist[i] = Utils.norm_mat_dist(Uinit * Vinit', Mtrue)
			if dist[i] <= eps
				return Uinit, Vinit, dist[1:i]
			end
            gradU[:], gradV[:] = bilinSubgrad(bProb, Uinit, Vinit)
			broadcast!(-, Uinit, Uinit, q * gradU / norm(gradU))
			broadcast!(-, Vinit, Vinit, q * gradV / norm(gradV))
            q = q * rho
        end
        return Uinit, Vinit, dist
    end


	"""
		pSgd(prob::MatCompProb, Xinit, iters; eta=1, eps=1e-10)

	Apply the projected subgradient method with Polyak step size to the matrix
	completion problem for `iters` iterations, given a problem instance `prob`
	and an initial estimate `Xinit`.
	"""
	function pSgd(prob::MatCompProb, Xinit, iters; eta=1, eps=1e-10)
		dist = fill(0.0, iters); grad = fill(0.0, size(Xinit))
		for i = 1:iters
			dist[i] = Utils.norm_mat_dist(Xinit * Xinit', prob.M)
			if dist[i] <= eps
				return Xinit, dist[1:i]
			end
			grad[:] = matCompSubgrad(prob, Xinit)
			step = eta * matCompLoss(prob, Xinit)
			broadcast!(-, Xinit, Xinit, step * grad / (norm(grad)^2))
		end
		return Xinit, dist
	end


    """
        pSgd_init(qProb::Union{SymQuadProb, QuadProb}, iters, delta; λ=1.0, rho=0.98)

    Apply the projected subgradient method with artificial "good"
    initialization to a quadratic sensing problem.
    """
    function pSgd_init(qProb::Union{SymQuadProb, QuadProb}, iters, delta;
					   λ=1.0, rho=0.98, eps=1e-10)
        Xtrue = qProb.X; d, r = size(Xtrue)
        randDir = randn(d, r); randDir = randDir / norm(randDir)
        Xinit = Xtrue + delta * randDir * norm(Xtrue)
        return pSgd(qProb, Xinit, iters, λ=λ, rho=rho, eps=eps)
    end


    """
        pSgd_init(bProb::BilinProb, iters, delta; λ=1.0, rho=0.98)

    Apply the projected subgradient method with artificial "good"
    initialization to a bilinear sensing problem.
    """
    function pSgd_init(bProb::BilinProb, iters, delta; λ=1.0, rho=0.98, eps=1e-10)
        Wtrue = bProb.W; Xtrue = bProb.X
		d1, r = size(Wtrue); d2, r = size(Xtrue)
		randW = randn(d1, r); randW /= norm(randW)
		randX = randn(d2, r); randX /= norm(randX)
		Uinit = Wtrue + delta * randW * norm(Wtrue)
		Vinit = Xtrue + delta * randX * norm(Xtrue)
        return pSgd(bProb, Uinit, Vinit, iters, λ=λ, rho=rho, eps=eps)
    end


	"""
		pSgd_init(prob::MatCompProb, iters, delta; eta=1.0, eps=1e-10)

	Apply the projected subgradient method with artifical "good" initialization
	to a PSD matrix completion problem, using the Polyak step size.
	"""
	function pSgd_init(prob::MatCompProb, iters, delta; eta=1.0, eps=1e-10)
        Xtrue = prob.X; d, r = size(Xtrue)
        randDir = randn(d, r); randDir = randDir / norm(randDir)
        Xinit = Xtrue + delta * randDir * norm(Xtrue)
        return pSgd(prob, Xinit, iters, eta=eta, eps=eps)
    end
end
