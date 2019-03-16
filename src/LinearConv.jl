"""
A module implementing nonsmooth optimization methods for a variety of
statistical problems that exhibit sharpness and weak convexity.
"""
module LinearConv

    include("Utils.jl")

    using LinearAlgebra
    using Random
    using Statistics


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
		S = Utils.genSparseMatrix(d, r, corr_lvl)
		W = X * X' + S
		return RpcaProb(W, X, S, 2 * Utils.abNorm(X, Inf, 2), corr_lvl)
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


	# TODO: Complete function
	function rpcaStep(prob::RpcaProb, Xk, C, gamma=10.0; maxIt=5000, eps=1e-10)
		Yk = copy(Xk); Ybest = copy(Xk); minF = Inf
		Gquad = fill(0.0, size(Xk))  # (sub)gradient 1
		Gnorm = fill(0.0, size(Xk))  # (sub)gradient 2
		for i = 1:maxIt
			# apply quadratic penalty gradient
			Utils.subg_sq21Norm(Yk, Xk, Gquad)
			Yk[:] = Yk[:] - (1 / gamma) * Gquad
			# TODO: apply gradient of ell-1 elementwise norm
			# apply projection
			Yk[:] = Utils.matProj_2inf(Yk, C)  # project back to C
		end
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
end
