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

#------
# structs
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


#--------
# Problem generators

	#= squared norm shortcut =#
	sqnorm(x) = norm(x)^2


	"""
		genSymQuadProb(d, m, r, noise_lvl=0.0; denseLvl=0.0)

	Generates a symmetrized quadratic sensing problem in dimensions
	``d \\times n`` where ``\\rank(X) = r`` with a desired noise level.
	If `denseLvl > 0`, adds dense gaussian noise.
	"""
	function genSymQuadProb(d, m, r, noise_lvl=0.0; denseLvl=0.0)
		A1 = randn(m, d); A2 = randn(m, d); X = randn(d, r)
		y = vec(mapslices(sqnorm, A1 * X, dims=[2]))
		y = y - vec(mapslices(sqnorm, A2 * X, dims=[2]))
		if denseLvl <= 1e-12
			Utils.corrupt_measurements!(y, noise_lvl, :gaussian)
		else
			# add dense noise here
			Utils.corrupt_measurements!(y, noise_lvl, :large_sparse)
			e = randn(length(y)); e = (svdvals(X)[r]^2) * e / norm(e)
			broadcast!(+, y, y, denseLvl * e)
		end
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
		genRpcaProb(d, r, corr_lvl=0.0; sparse_signal=nothing)

	Generates a robust pca problem in dimensions ``d \\times r`` where
	a `corr_lvl` fraction of entries are outliers. If `sparse_signal` is set,
	treats it as the sparse corruption instead.
	"""
	function genRpcaProb(d, r, corr_lvl=0.0; sparse_signal=nothing)
		if sparse_signal != nothing
			if isa(sparse_signal, Array{<:Number})
				if size(sparse_signal) != (d, d)
					throw(Exception("Size of sparse_signal incorrect!"))
				end
			end
		end
		X = Utils.genIncoherentMatrix(d, r)
		S = Utils.genSparseMatrix(d, d, corr_lvl, sMat=sparse_signal)
		W = X * X' + S
		return RpcaProb(W, X, S, 2 * Utils.abNorm(X, Inf, 2), corr_lvl)
	end


	"""
		genMatCompProb(d, r, sample_freq=1.0; denseLvl=0.0)

	Generate a matrix completion problem in dimensions ``d \\times r`` where
	elements are sampled with `sample_freq` frequency.
	If `denseLvl > 0`, adds dense Gaussian noise.
	"""
	function genMatCompProb(d, r, sample_freq=1.0; denseLvl=0.0)
		X = Utils.genIncoherentMatrix(d, r)
		mask = Utils.genMask(d, sample_freq)
		svdObj = svd(X); S = svdObj.S
		S[:] = ones(length(S)) * median(S); X = svdObj.U * (S .* svdObj.Vt)
		if denseLvl <= 1e-12
			# equalize singular values according to paper
			return MatCompProb(X, X * X', mask, sample_freq)
		else
			# equalize singular values according to paper
			e = randn(d, d); e = sqrt(sample_freq) * (e / norm(e)) * (median(S)^2)
			return MatCompProb(X, X * X' .+ e * denseLvl, mask, sample_freq)
		end
	end

#-------
# Residuals and losses

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


	function symQuadNaiveLoss(qProb, Xcurr)
		r = symQuadRes(qProb, Xcurr)
		return (2 / length(r)) * (norm(r)^2)
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


	function bilinNaiveLoss(bProb, Ucurr, Vcurr)
		r = bilinRes(bProb, Ucurr, Vcurr)
		return (2 / length(r)) * (norm(r)^2)
	end


	"""
		matCompLoss(prob::MatCompProb, Xcurr)

	Returns the Frobenius error of the PSD matrix completion iterate.
	"""
	function matCompLoss(prob::MatCompProb, Xcurr)
		return norm(prob.mask .* (Xcurr * Xcurr' - prob.M))
	end


	function matCompNaiveLoss(prob::MatCompProb, Xcurr)
		return norm(prob.mask .* (Xcurr * Xcurr' - prob.M))^2
	end


#-------
# Quadratic sensing

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
	function symQuadNaiveGD(qProb, Xcurr, iters, sStep;
		                    eps=1e-12, use_polyak=false)
		# note: sStep is either a callable or a Number
		stepSize = Utils.setupStep(sStep)
		dist = fill(0.0, iters); Xtrue = qProb.X
		for k = 1:iters
			dist[k] = Utils.norm_mat_dist(Xcurr * Xcurr', Xtrue * Xtrue')
			if dist[k] <= eps
				return Xcurr, dist[1:k]
			end
			if use_polyak
				grad = symQuadGrad(qProb, Xcurr)
				err = symQuadNaiveLoss(qProb, Xcurr)
				broadcast!(-, Xcurr, Xcurr, stepSize(k) * err * grad / (norm(grad)^2))
			else
				broadcast!(-, Xcurr, Xcurr, stepSize(k) * symQuadGrad(qProb, Xcurr))
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
	function symQuadNaiveGD_init(qProb, delta, iters, sStep;
		                         eps=1e-12, use_polyak=false)
		Xtrue = qProb.X; d, r = size(Xtrue)
        randDir = randn(d, r); randDir = randDir / norm(randDir)
        Xinit = Xtrue + delta * randDir * norm(Xtrue)
		return symQuadNaiveGD(qProb, Xinit, iters, sStep, eps=eps,
		                      use_polyak=use_polyak)
	end


	function _mk_XMat(qProb, Xk)
		m = length(qProb.y)
		return 2 .* vcat((kron(qProb.A1[i, :]' * Xk, qProb.A1[i, :]') - kron(
			qProb.A2[i, :]' * Xk, qProb.A2[i, :]') for i=1:m)...)
	end


	function symQuadProxStep(qProb, Xk, Zinit, Zsol, Znew, Yinit, Ysol, Ynew,
		Lk, Nk, Lsys, Lfact; γ=5, maxIt=2500, ρ=10, ϵ=1e-3)
		# sizes to retrieve
		szX = prod(size(qProb.X))
		# store [vec(Uk); vec(Vk)]
		Zk = vec(Xk); m = length(qProb.y); n = length(Zk)
		c = vec(mapslices(sqnorm, qProb.A1 * Xk, dims=[2])) .- vec(
			mapslices(sqnorm, qProb.A2 * Xk, dims=[2])) + qProb.y
		A = _mk_XMat(qProb, Xk)  # => A * [vec(U); vec(V)]
		Lsys[:] = 1.0I + A' * A   # linear system
		Lfact[:] = cholesky(Lsys).L   # cached cholesky factor
		Zinit[:] = Zk; Yinit[:] = A * Zk;
		fill!(Lk, 0.0); fill!(Nk, 0.0)
		for i = 1:maxIt
			# Basic updates
			Zsol[:] = 1 / ((1/γ) + ρ) * (ρ * (Zinit - Lk) + (1/γ) * Zk)
			Ysol[:] = c + soft_thres(Yinit - Nk - c, 1/(ρ * m))
			Vnew = (Zsol + Lk) + A' * (Ysol + Nk)
			Znew[:] = Lfact' \ (Lfact \ Vnew)  # backsolve
			Ynew[:] = A * Znew
			broadcast!(+, Lk, Lk, Zsol - Znew)
			broadcast!(+, Nk, Nk, Ysol - Ynew)
			res_p = norm(vcat(Znew - Zsol, Ynew - Ysol))
			res_d = ρ * norm(vcat(Zinit - Znew, Yinit - Ynew))
			eps_p = ϵ * (sqrt(n) + max(norm(Znew), norm(Ynew)))
			eps_d = ϵ * (sqrt(n) + max(norm(Lk), norm(Nk)))
			if (res_p < eps_p) && (res_d < eps_d)
				break;
			else  # update estimates
				Zinit[:] = Znew; Yinit[:] = Ynew
			end
		end
		return reshape(Znew, size(Xk))
	end


	"""
		symQuadProxlin(qProb, Xinit, iters, γ=5; maxIt=2000, ρ=nothing,
					   ϵ=(i -> min(1e-4, 4.0^(-i))), eps=1e-10)

	Run the prox-linear method for a symmetrized quadratic sensing problem
	with initial estimate `Xinit` for `iters` iterations and prox-parameter
	`γ`.
	"""
	function symQuadProxlin(qProb, Xinit, iters, γ=5; maxIt=500, ρ=10,
		                    ϵ=1e-3, eps=1e-10)
		Mtrue = qProb.X * qProb.X'
		step = Utils.setupStep(ϵ)  # setup step size
		# preallocate everything
		Zinit = fill(0.0, size(vec(Xinit)))
		Znew = copy(Zinit); Zsol = copy(Znew)
		Yinit = fill(0.0, length(qProb.y))
		Ynew = copy(Yinit); Ysol = copy(Ynew)
		Lk = copy(Zinit); Nk = copy(Yinit)
		A = _mk_XMat(qProb, Xinit)
		Lsys = 1.0I + A' * A   # linear system
		Lfact = cholesky(Lsys).L   # cached cholesky factor
		dists = fill(0.0, iters)
		for k = 1:iters
			dists[k] = norm(Xinit * Xinit' - Mtrue) / norm(Mtrue)
			if dists[k] <= eps
				return Xinit, dists[1:k]
			end
			# solve a proximal subproblem
			Xinit[:] = symQuadProxStep(
				qProb, Xinit, Zinit, Zsol, Znew, Yinit, Ysol, Ynew, Lk, Nk,
				Lsys, Lfact, γ=γ, ϵ=step(k), ρ=ρ, maxIt=maxIt)
		end
		return Xinit, dists
	end


	"""
		symQuadProxlin_init(bProb, delta, iters; γ=5, maxIt=2000, ρ=nothing,
					        ϵ=(i -> min(1e-4, 4.0^(-i))), eps=1e-10)

	Run the prox-linear method for a symmetrized quadratic sensing problem
	starting ``\\delta``-close to the optimal solution for `iters` iterations.
	"""
	function symQuadProxlin_init(qProb, delta, iters; γ=5, maxIt=2000,
		                         ρ=nothing, ϵ=(i -> min(1e-4, 4.0^(-i))),
								 eps=1e-10)
		if (ρ == nothing)
			ρ = 1 / length(qProb.y)
		end
		Xtrue = qProb.X
		d, r = size(Xtrue); randX = randn(d, r); randX /= norm(randX)
		Xinit = Xtrue + delta * randX * norm(Xtrue)
		return symQuadProxlin(qProb, Xinit, iters, γ, maxIt=maxIt, ρ=ρ, ϵ=ϵ,
		                      eps=eps)
	end


#-------
# Bilinear sensing

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
	function bilinNaiveGD(bProb, Ucurr, Vcurr, iters, sSize;
		                  eps=1e-12, use_polyak=false)
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
			if use_polyak
				err = bilinNaiveLoss(bProb, Ucurr, Vcurr)
				broadcast!(-, Ucurr, Ucurr, err * stepSize(i) * gradU / (norm(gradU)^2))
				broadcast!(-, Vcurr, Vcurr, err * stepSize(i) * gradV / (norm(gradV)^2))
			else
				broadcast!(-, Ucurr, Ucurr, stepSize(i) * gradU)
				broadcast!(-, Vcurr, Vcurr, stepSize(i) * gradV)
			end
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
	function bilinNaiveGD_init(bProb, delta, iters, sSize;
		                       eps=1e-12, use_polyak=false)
		Wtrue = bProb.W; Xtrue = bProb.X
		d1, r = size(Wtrue); d2, r = size(Xtrue)
		randW = randn(d1, r); randW /= norm(randW)
		randX = randn(d2, r); randX /= norm(randX)
		Uinit = Wtrue + delta * randW * norm(Wtrue)
		Vinit = Xtrue + delta * randX * norm(Xtrue)
        return bilinNaiveGD(bProb, Uinit, Vinit, iters, sSize,
		                    eps=eps, use_polyak=use_polyak)
	end


	"""
		_mkUVMats(bProb, Uk, Vk)

	Make a large sparse block matrix for the graph-splitting ADMM subproblems.
	"""

	function _mk_UVMats(bProb, Uk, Vk)
		m = length(bProb.y)
		return hcat(
			vcat((kron(bProb.B[i, :]' * Vk, bProb.A[i, :]') for i=1:m)...),
			vcat((kron(bProb.A[i, :]' * Uk, bProb.B[i, :]') for i=1:m)...))
	end


	#= Implements one proximal step for bilinear prox-linear method =#
	function bilinProxStep(bProb, Uk, Vk, Zinit, Zsol, Znew, Yinit, Ysol, Ynew,
		Lk, Nk, Lsys, Lfact; γ=5, maxIt=2500, ρ=10, ϵ=1e-3)
		# sizes to retrieve
		szU = prod(size(bProb.W)); szV = prod(size(bProb.X))
		# store [vec(Uk); vec(Vk)]
		Zk = vcat(vec(Uk), vec(Vk)); m = length(bProb.y); n = length(Zk)
		A = _mk_UVMats(bProb, Uk, Vk)  # => A * [vec(U); vec(V)]
		c = Utils.rowwise_prod(bProb.A * Uk, bProb.B * Vk) + bProb.y
		Lsys[:] = 1.0I + A' * A   # linear system
		Lfact[:] = cholesky(Lsys).L   # cached cholesky factor
		Zinit[:] = Zk; Yinit[:] = A * Zk
		fill!(Lk, 0.0); fill!(Nk, 0.0)
		for i = 1:maxIt
			# Basic updates
			Zsol[:] = 1 / ((1/γ) + ρ) * (ρ * (Zinit - Lk) + (1/γ) * Zk)
			Ysol[:] = c + soft_thres(Yinit - Nk - c, 1/(ρ * m))
			Vnew = (Zsol + Lk) + A' * (Ysol + Nk)
			Znew[:] = Lfact' \ (Lfact \ Vnew)  # backsolve
			Ynew[:] = A * Znew
			broadcast!(+, Lk, Lk, Zsol - Znew)
			broadcast!(+, Nk, Nk, Ysol - Ynew)
			res_p = norm(vcat(Znew - Zsol, Ynew - Ysol))
			res_d = ρ * norm(vcat(Zinit - Znew, Yinit - Ynew))
			eps_p = ϵ * (sqrt(n) + max(norm(Znew), norm(Ynew)))
			eps_d = ϵ * (sqrt(n) + max(norm(Lk), norm(Nk)))
			if (res_p < eps_p) && (res_d < eps_d)
				break;
			else  # update estimates
				Zinit[:] = Znew; Yinit[:] = Ynew
			end
		end
		Unew = reshape(Znew[1:szU], size(Uk))
		Vnew = reshape(Znew[(szU+1):end], size(Vk))
		return Unew, Vnew
	end


	"""
		bilinProxlin(bProb, Uinit, Vinit, iters, γ=5; maxIt=500, ρ=10,
		             ϵ=1e-3, eps=1e-10)

	Run the prox-linear method for a bilinear sensing problem with initial
	estimates `Uinit, Vinit` for `iters` iterations and prox-parameter
	γ.
	"""
	function bilinProxlin(bProb, Uinit, Vinit, iters, γ=5; maxIt=500, ρ=10,
		                  ϵ=1e-3, eps=1e-10)
		Mtrue = bProb.W * bProb.X'
		step = Utils.setupStep(ϵ)  # setup step size
		# preallocate everything
		Zinit = fill(0.0, size(vcat(vec(Uinit), vec(Vinit))))
		Znew = copy(Zinit); Zsol = copy(Znew)
		Yinit = fill(0.0, length(bProb.y))
		Ynew = copy(Yinit); Ysol = copy(Ynew)
		Lk = copy(Zinit); Nk = copy(Yinit)
		A = _mk_UVMats(bProb, Uinit, Vinit)  # => A * [vec(U); vec(V)]
		Lsys = 1.0I + A' * A   # linear system
		Lfact = cholesky(Lsys).L   # cached cholesky factor
		dists = fill(0.0, iters)
		for k = 1:iters
			dists[k] = norm(Uinit * Vinit' - Mtrue) / norm(Mtrue)
			if dists[k] <= eps
				return Uinit, Vinit, dists[1:k]
			end
			# solve a proximal subproblem
			Uinit[:], Vinit[:] = bilinProxStep(
				bProb, Uinit, Vinit, Zinit, Zsol, Znew, Yinit, Ysol, Ynew,
				Lk, Nk, Lsys, Lfact, γ=γ, ϵ=step(k), ρ=ρ, maxIt=maxIt)
		end
		return Uinit, Vinit, dists
	end


	"""
		bilinProxlin_init(bProb, delta, iters; γ=5, maxIt=2000, ρ=nothing,
					      ϵ=(i -> min(1e-4, 4.0^(-i))), eps=1e-10)

	Run the prox-linear method for a bilinear sensing problem starting
	``\\delta``-close to the optimal solution for `iters` iterations.
	"""
	function bilinProxlin_init(bProb, delta, iters; γ=5, maxIt=2000, ρ=nothing,
		                       ϵ=(i -> min(1e-4, 4.0^(-i))), eps=1e-10)
		if (ρ == nothing)
			ρ = 1 / length(bProb.y)
		end
		Wtrue = bProb.W; Xtrue = bProb.X
		d1, r = size(Wtrue); d2, r = size(Xtrue)
		randW = randn(d1, r); randW /= norm(randW)
		randX = randn(d2, r); randX /= norm(randX)
		Uinit = Wtrue + delta * randW * norm(Wtrue)
		Vinit = Xtrue + delta * randX * norm(Xtrue)
		return bilinProxlin(bProb, Uinit, Vinit, iters, γ, maxIt=maxIt,
		                    ρ=ρ, ϵ=ϵ, eps=eps)
	end


#--------------------------------------
# Matrix completion

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
	function matCompNaiveGD(prob::MatCompProb, Xcurr, iters, sSize;
							eps=1e-12, use_polyak=false)
		stepSize = Utils.setupStep(sSize)
		dist = fill(0.0, iters)
		for i = 1:iters
			dist[i] = Utils.norm_mat_dist(Xcurr * Xcurr', prob.M)
			if dist[i] <= eps
				return Xcurr, dist[1:i]
			end
			if use_polyak
				grad = matCompGrad(prob, Xcurr)
				err = matCompNaiveLoss(prob, Xcurr)
				broadcast!(-, Xcurr, Xcurr, stepSize(i) * err * grad / (norm(grad)^2))
			else
				broadcast!(-, Xcurr, Xcurr, stepSize(i) * matCompGrad(prob, Xcurr))
			end
		end
		return Xcurr, dist
	end


	"""
		matCompNaiveGD_init(prob::MatCompProb, delta, iters, sSize; eps=1e-12)

	Solve a PSD matrix completion problem using "naive" gradient descent on a
	squared Frobenius formulation, given a step size `sSize` which is either a
	number or a callable accepting the iteration number as its argument.
	Starts with an iterate `delta`-close to the ground truth.
	"""
	function matCompNaiveGD_init(prob::MatCompProb, delta, iters, sSize;
                                 eps=1e-12, use_polyak=false)
		d, r = size(prob.X); rDir = randn(d, r); rDir /= norm(rDir)
		Xinit = prob.X + delta * norm(prob.X) * rDir
		return matCompNaiveGD(prob, Xinit, iters, sSize,
		                      eps=eps, use_polyak=use_polyak)
	end


	"""
		matCompInternalSolver(prob::MatCompProb, W, Xcurr, maxIter, ϵ, ρ; eps = 1e-12)

	A graph splitting-based ADMM solver for the matrix completion prox-linear
	method.
	"""
	function matCompInternalSolver(prob::MatCompProb, W, Xcurr, maxIter, ϵ, ρ; eps = 1e-12)
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
		Diff = prob.M - Xcurr * Xcurr'
		y = Diff[prob.mask .> 0]
		XkI = kron(Xcurr, zeros(d,d) + UniformScaling(1))
		A = W*XkI

		Q = zeros(d*r, d*r) + UniformScaling(1) + A'*A
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


	"""
		matCompProxLinear(prob::MatCompProb, Xcurr, maxIter, ϵ; eps=1e-12)

	Solve a matrix completion problem instance from a starting iterate `Xcurr`
	for `maxIter` iterations with a penalty parameter ``\\epsilon``.
	"""
	function matCompProxLinear(prob::MatCompProb, Xcurr, maxIter, ϵ; eps = 1e-12)
		dist = fill(0.0, maxIter)
		d, r = size(Xcurr)
		idx = findall(x->(x.>0),prob.mask)
		m = nnz(sparse(prob.mask))
		k1 = zeros(Int64, m, 1)
		k2 = zeros(Int64, m, 1)
		for ii = 1:m
			k1[ii] = LinearIndices(prob.mask)[idx[ii]]
			k2[ii] = LinearIndices(prob.mask)[CartesianIndex(idx[ii][2], idx[ii][1])]
		end
		W = sparse([Array(1:m); Array(1:m)], vec([k1; k2]), ones(2*m), m, d^2);
		Mtrue = prob.X * prob.X'   # just in case we have dense gaussian noise
		for ii = 1:maxIter
			dist[ii] = Utils.norm_mat_dist(Xcurr * Xcurr', Mtrue)
			if dist[ii] <= eps
				return Xcurr, dist[1:ii]
			end
			Xcurr = matCompInternalSolver(prob, W, Xcurr, maxIter, ϵ, 1/m)
		end
		return Xcurr, dist
	end


	"""
		matCompProxLinear_init(prob, delta, maxIter; ϵ=1.0, eps=1e-12)

	Solve a matrix completion instance using the prox-linear method starting
	at an iterate that is `delta`-close to the ground truth.
	"""
	function matCompProxLinear_init(prob::MatCompProb, maxIter, delta; ϵ=1.0, eps=1e-12)
		Xtrue = prob.X; d, r = size(Xtrue)
		randDir = randn(d, r); randDir = randDir / norm(randDir)
		Xinit = Xtrue + delta * randDir * norm(Xtrue)
		return matCompProxLinear(prob, Xinit, maxIter, ϵ, eps=eps)
	end


#-----
# RPCA

	#= Helper function for the cost of the prox-linear objective for RPCA =#
	proxlin_rpca_cost(X, Xk, C, gamma) = begin
		fcA = sum(abs.(X * Xk' + Xk * X' - C))
		fcB = (1 / (2 * gamma)) * Utils.abNorm(X - Xk, 1, 2)
		return fcA + fcB
	end


	"""
		rpca_subgrad(prob::RpcaProb, Xk)

	Return a subgradient of the non-euclidean rPCA problem at `Xk`.
	"""
	function rpca_subgrad(prob::RpcaProb, Xk)
		return sign.(Xk * Xk' - prob.W) * Xk
	end


	"""
		rpca_subgrad_euc(prob::RpcaProb, Xk, Sk)

	Return a subgradient of the Euclidean rPCA problem at `(Xk, Sk)`.
	"""
	function rpca_subgrad_euc(prob::RpcaProb, Xk, Sk)
		res = Xk * Xk' + Sk - prob.W
		return (res * Xk + res' * Xk) / norm(res), res / norm(res)
	end


	"""
		ell21_prox(Xk, Z, Ztilde, C, ρ; gamma=10.0, maxIt=1000)

	Computes the proximal operator of the ``\\ell_{2,1}`` norm of a matrix
	by minimizing
	``
		\\frac{1}{2 \\gamma_k} \\| X - X_k \\|_{2,1}^2
		+ \\frac{\\rho}{2} \\| X - (Z - \\tilde{Z}) \\|_F^2
	``
	"""
	function ell21_prox(Xk, Zk, Lk, rho; gamma=10.0)
		# f(x) = g(ax + b) -> prox_{λ f}(v) = prox_{λ g}(av + b) - b
		# set b = -Xk to get below result
		return Utils.prox_sq21Norm(Zk - Lk - Xk, 1 / (gamma * rho)) + Xk
	end


	soft_thres(x, alpha) = sign.(x) .* max.(abs.(x) .- alpha, 0)


	"""
		matEll1_prox(YvecK, c, nuK, rho)

	Computes the proximal operator for the matrix elementwise ell-1 norm given
	the vectorized matrix YvecK.
	"""
	function matEll1_prox(YvecK, c, nuK, rho)
		# soft thresholding operator for elemwise 1-norm
		return c + soft_thres(YvecK - nuK - c, 1/rho)
	end


	"""
		rpcaProxlinStep(prob, Xk, C, iters; gamma=10.0, gradIt=1000)

	Apply the graph-splitting ADMM algorithm to perform one step of the
	prox-linear algorithm applied to robust PCA, linearized around `Xk` and
	under a ``\\ell_{2, \\infty}`` norm constraint given by `C`.
	"""
    function rpcaProxlinStep(prob::RpcaProb, Xk, C, iters,
		                     Lk, Nk, Zinit, Znew, Zsol, Yinit, Ynew, Ysol,
							 Vnew, newSol, Lsys, Lfact, c, A;
							 K=nothing, eps=1e-5, gamma=10.0, rho=5)
		n, r = size(Xk); zsSize = n * r
		# linear transformation for ell1-subproblem
		A[:] = kron(Xk, sparse(1.0I, n, n)) + kron(sparse(1.0I, n, n), Xk) * K
		# PSD matrix for projection step
		Lsys[:] = UniformScaling(1) + A' * A
		Lfact[:] = sparse(cholesky(Lsys).L)
		c[:] = vec(Xk * Xk' + prob.W)
		# primal/dual variables
		fill!(Zinit, 0.0); fill!(Yinit, 0.0)
		fill!(Lk, 0.0); fill!(Nk, 0.0)
		for i = 1:iters
			# get proximal operators
			Znew[:] = ell21_prox(Xk, Zinit, Lk, rho, gamma=gamma)
			Ynew[:] = matEll1_prox(Yinit, c, Nk, rho)
			# copyto!(Zinit, Znew); copyto!(Yinit, Ynew)
			# set the vector for graph splitting (rest is zeros)
			Vnew[:] = vec(Znew + Lk) + A' * (Ynew + Nk)
			# update Znew by solving the linear system
			newSol[:] = Lfact' \ (Lfact \ Vnew)
			Zsol[:] = reshape(newSol, n, r)
			# upate Ynew by multiplying
			Ysol[:] = A * newSol
			# update dual variables
			broadcast!(+, Lk, Lk, Znew - Zsol)
			broadcast!(+, Nk, Nk, Ynew - Ysol)
			# compute residuals
			resP = norm(Zsol - Znew)^2 + norm(Ysol - Ynew)^2
			resD = (rho^2) * (norm(Zsol - Zinit)^2 + norm(Ysol - Yinit)^2)
			if (resP < (eps * (sqrt(n * r) + max(norm(Zinit), norm(Yinit))))^2) &&
			   (resD < (eps * (sqrt(n * r) + max(norm(Lk), norm(Nk))))^2)
				return Zsol
			else
				copyto!(Zinit, Zsol); copyto!(Yinit, Ysol)
			end
		end
		return Zsol
	end


	"""
		rpcaProxLin(prob::RpcaProb, Xk, C, iters; eps=1e-12, gamma=10.0, maxIt=1000)

	Solve a robust PCA instance using the prox-linear method for `iters`
	iterations, given an upper bound `C` on the ``2,\\infty`` norm of the
	solution. The parameter ``\\gamma`` is the scale of the norm penalty for
	the proximal subproblems, while `maxIt` controls the
	maximum number of iterations per subproblem.
	"""
	function rpcaProxLin(prob::RpcaProb, Xk, C, iters;
						 eps=1e-8, gamma=10.0, maxIt=1000, rho=0.1,
						 inner_eps=1e-3)
		Yk = copy(Xk); dists = fill(0.0, iters); M = prob.X * prob.X'
		K = Utils.commutator(size(Xk)...)
		n, r = size(Xk)
		# initialize variables
		# linear transformation for ell1-subproblem
		A = kron(Xk, sparse(1.0I, n, n)) + kron(sparse(1.0I, n, n), Xk) * K
		# PSD matrix for projection step
		Lsys = UniformScaling(1) + A' * A
		Lfact = sparse(cholesky(Lsys).L)
		c = vec(Xk * Xk' + prob.W)
		# dual variables
		Lk = fill(0.0, (n, r)); Nk = fill(0.0, length(c))
		# primal variables
		Zinit = fill(0.0, size(Xk)...); Yinit = fill(0.0, size(A * vec(Xk))...)
		Znew = copy(Zinit); Ynew = copy(Yinit)
		Zsol = copy(Zinit); Ysol = copy(Yinit)
		Vnew = fill(0.0, n * r)
		# vector to solve for in graph projection step
		newSol = fill(0.0, n * r)
		for i = 1:iters
			dists[i] = norm(Yk * Yk' - M, 1) / norm(M, 1)
			if dists[i] <= eps
				return Yk, dists[1:i]
			end
			# perform a prox-step
			Yk[:] = rpcaProxlinStep(prob, Yk, C, maxIt, Lk, Nk,
			                        Zinit, Znew, Zsol, Yinit, Ynew, Ysol,
									Vnew, newSol, Lsys, Lfact, c, A,
									rho=rho, K=K, eps=(inner_eps / i))
		end
		return Yk, dists
	end


	"""
		rpcaProxLin_init(prob, iters, delta; eps=1e-8, gamma=10.0, inner_eps=1e-3,
						 maxIt=500, rho=nothing)

	Solve a robust PCA instance using the prox-linear method for `iters`
	iterations, starting at a point ``\\delta``-close to the ground truth.
	"""
	function rpcaProxLin_init(prob::RpcaProb, iters, delta;
							  eps=1e-8, gamma=10.0, inner_eps=1e-3, maxIt=500,
							  rho=5)
		Xtrue = prob.X; d, r = size(Xtrue)
		randDir = randn(d, r); randDir = randDir / norm(randDir)
		Xinit = Xtrue + delta * randDir * norm(Xtrue)
		Xnorm = Utils.abNorm(Xtrue, Inf, 2)
		return rpcaProxLin(prob, Xinit, 2 * Xnorm, iters,
						   eps=eps, gamma=gamma, maxIt=maxIt, rho=rho,
						   inner_eps=inner_eps)
	end

#------
# Subgradient wrappers

    """
        pSgd(qProb::QuadProb, Xinit, iters; λ = 1.0, rho = 0.98)

    Apply the projected subgradient method with geometrically decaying step
    to the quadratic sensing problem for `iters` iterations, given a problem
    instance `qProb` and an initial estimate `Xinit`.
    """
    function pSgd(qProb::QuadProb, Xinit, iters; λ=1.0, rho = 0.98, eps=1e-10)
        Xtrue = qProb.X; d, r = size(Xtrue); grad = fill(0.0, (d, r))
        q = λ; dist = fill(0.0, iters);
        for i = 1:iters
            dist[i] = Utils.norm_mat_dist(Xinit * Xinit', Xtrue * Xtrue')
			if dist[i] <= eps
				return Xinit, dist[1:i]
			end
            grad[:] = quadSubgrad(qProb, Xinit)
			if qProb.pfail <= 1e-10
				step = λ * quadRobustLoss(qProb, Xinit)
				broadcast!(-, Xinit, Xinit, step * grad / (norm(grad)^2))
			else
				broadcast!(-, Xinit, Xinit, q * grad / norm(grad))
            	q = q * rho
			end
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
			if qProb.pfail <= 1e-10
				step = λ * symQuadRobustLoss(qProb, Xinit)
				broadcast!(-, Xinit, Xinit, step * grad / (norm(grad)^2))
			else
				broadcast!(-, Xinit, Xinit, q * grad / norm(grad))
            	q = q * rho
			end
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
			if bProb.pfail <= 1e-10
				# need slightly smaller step for Polyak with bilinear
				step = 0.5 * λ * bilinRobustLoss(bProb, Uinit, Vinit)
				broadcast!(-, Uinit, Uinit, step * gradU / (norm(gradU)^2))
				broadcast!(-, Vinit, Vinit, step * gradV / (norm(gradV)^2))
			else
				broadcast!(-, Uinit, Uinit, q * gradU / norm(gradU))
				broadcast!(-, Vinit, Vinit, q * gradV / norm(gradV))
            	q = q * rho
			end
        end
        return Uinit, Vinit, dist
    end


	"""
		pSgd(prob::RpcaProb, Xinit, Sinit, iters; λ=1.0, rho=0.98, eps=1e-10)

	Apply the projected subgradient method with geometrically decaying step
	size to a robust PCA problem instance, starting from `Xinit` for `iters`
	iterations.
	"""
	function pSgd_euc(prob::RpcaProb, Xinit, Sinit, iters;
                      λ=1.0, rho=0.98, eps=1e-10)
		q = λ; dist = fill(0.0, iters); M = prob.X * prob.X'
		gradX = fill(0.0, size(Xinit)...); gradS = fill(0.0, size(Sinit)...)
		for i = 1:iters
			dist[i] = Utils.norm_mat_dist(Xinit * Xinit', M)
			if dist[i] <= eps
				return Xinit, dist[1:i]
			end
			gradX, gradS = rpca_subgrad_euc(prob, Xinit, Sinit)
			if prob.pfail <= 1e-10  # use Polyak step size
				cost = norm(Xinit * Xinit' + Sinit - prob.W)
				step = 0.75 * λ * cost
				broadcast!(-, Xinit, Xinit, step * gradX / (norm(gradX)^2))
				broadcast!(-, Sinit, Sinit, step * gradS / (norm(gradS)^2))
			else
				broadcast!(-, Xinit, Xinit, q * gradX / norm(gradX))
				broadcast!(-, Sinit, Sinit, q * gradS / norm(gradS))
			end
			# project to {2,∞} ball
			Xinit[:] = Utils.matProj_2inf(Xinit, prob.gamma)
			# project to l1-norm ball
			@inbounds for k = 1:(size(Sinit)[1])
				Sinit[k, :] = Utils.l1Proj(Sinit[k, :], norm(prob.S[k, :], 1))
			end
			q = q * rho
		end
		return Xinit, Sinit, dist
	end


	"""
		pSgd(prob::RpcaProb, Xinit, iters; λ=1.0, rho=0.98, eps=1e-10)

	Apply the projected subgradient method with geometrically decaying step
	size to a robust PCA problem instance, starting from `Xinit` for `iters`
	iterations, using the non-Euclidean formulation.
	"""
	function pSgd_nEuc(prob::RpcaProb, Xinit, iters;
                       λ=1.0, rho=0.98, eps=1e-10)
		q = λ; dist = fill(0.0, iters); M = prob.X * prob.X'
		gradX = fill(0.0, size(Xinit)...)
		for i = 1:iters
			dist[i] = norm(Xinit * Xinit' - M, 1) / norm(M, 1)
			if dist[i] <= eps
				return Xinit, dist[1:i]
			end
			gradX = rpca_subgrad(prob, Xinit)
			if prob.pfail <= 1e-10  # use Polyak step size
				cost = norm(Xinit * Xinit' - prob.W, 1)
				step = 0.75 * λ * cost
				broadcast!(-, Xinit, Xinit, step * gradX / (norm(gradX)^2))
			else
				broadcast!(-, Xinit, Xinit, q * gradX / norm(gradX))
			end
			# project to {2,∞} ball
			Xinit[:] = Utils.matProj_2inf(Xinit, prob.gamma)
			q = q * rho
		end
		return Xinit, dist
	end


	"""
		pSgd(prob::MatCompProb, Xinit, iters; eta=1, eps=1e-10)

	Apply the projected subgradient method with Polyak step size to the matrix
	completion problem for `iters` iterations, given a problem instance `prob`
	and an initial estimate `Xinit`.
	"""
	function pSgd(prob::MatCompProb, Xinit, iters; eta=1, eps=1e-10)
		dist = fill(0.0, iters); grad = fill(0.0, size(Xinit))
		Mtrue = prob.X * prob.X'
		for i = 1:iters
			dist[i] = Utils.norm_mat_dist(Xinit * Xinit', Mtrue)
			if dist[i] <= eps
				return Xinit, dist[1:i]
			end
			grad[:] = matCompSubgrad(prob, Xinit)
			step = eta * matCompLoss(prob, Xinit)
			broadcast!(-, Xinit, Xinit, step * grad / (norm(grad)^2))
		end
		return Xinit, dist
	end

#-------
# PSGD with init

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
		pSgd_init(prob::RpcaProb, iters, delta; λ=1.0, rho=0.98, eps=1e-10)

	Apply the projected subgradient method with artificial "good" initialization
	to a robust PCA problem.
	"""
	function pSgd_init(prob::RpcaProb, iters, delta; λ=1.0, rho=0.98, eps=1e-10,
		               mode=:euclidean)
		Xtrue = prob.X; d, r = size(Xtrue)
		randDir = randn(d, r); randDir = randDir / norm(randDir)
		Xinit = Xtrue + delta * randDir * norm(Xtrue)
		if mode == :euclidean
			randSDir = randn(d, d); randSDir = randSDir / norm(randSDir)
			Sinit = prob.S + delta * randSDir * norm(prob.S)
			# project
			@inbounds for i = 1:d
				Sinit[i, :] = Utils.l1Proj(Sinit[i, :], norm(prob.S[i, :], 1))
			end
			return pSgd_euc(prob, Xinit, Sinit, iters, λ=λ, rho=rho, eps=eps)
		else
			return pSgd_nEuc(prob, Xinit, iters, λ=λ, rho=rho, eps=eps)
		end
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
