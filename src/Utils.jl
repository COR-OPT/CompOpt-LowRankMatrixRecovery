module Utils

    using FFTW
    using LinearAlgebra
    using Random
    using Statistics

    """
        genMtx(d, n, r)

    Generates a rank-r matrix of size d x n.
    """
    function genMtx(d, n, r)
        # get U, V factors - keep correct dimensions
        U_fact = qr(randn(d, d)).Q; V_fact = qr(randn(n, n)).Q
        U_fact = U_fact[1:end, 1:r]; V_fact = V_fact[1:r, 1:end]
        D = 0.1 .+ randn(r).^2  # prevent zero numerical rank
        return U_fact * (D .* V_fact)
    end


    """
        corrupt_measurements!(y, noise_lvl, noise=:gaussian)

    Corrupt the set of measurements ``y`` with noise of a specified level and
    type.
    """
    function corrupt_measurements!(y, noise_lvl, noise=:gaussian)
        m = length(y); num_corr = trunc.(Int, noise_lvl * m)
        if noise == :gaussian
            noise = randn(num_corr)
            y[randperm(m)[1:num_corr]] += noise
        elseif noise == :sq_gaussian  # squared gaussian
            noise = randn(num_corr) .^ 2
            y[randperm(m)[1:num_corr]] += noise
        end
    end


    """
        ortho_dist(A, B)

    Calculates the distance between matrices ``A, B`` up to a natural rotation
    by solving ``\\min_{Q \\in O(d)} \\| A - B Q \\|_F``.
    """
    function ortho_dist(A, B)
        Msvd = svd(B' * A); Q = Msvd.U * Msvd.Vt;
        return norm(A - B * Q)
    end


    """
        rowwise_prod(A, B)

    Computes the dot product between the rows of matrices ``A`` and ``B``.
    """
    function rowwise_prod(A, B)
        d1, r1 = size(A); d2, r2 = size(B)
        @assert (d1 == d2) && (r1 == r2)
        result = fill(0.0, d1)
        @inbounds for i = 1:d1
            result[i] = (A[i, :])' * B[i, :]
        end
        return result
    end


    """
        norm_mat_dist(A, B)

    Computes the normalized matrix distance (in Frobenius norm) between
    matrices ``A`` and ``B``.
    """
    function norm_mat_dist(A, B)
        return sqrt(
            abs((norm(A)^2 + norm(B)^2 - 2 * dot(A, B)) / norm(B)^2))
    end


    """
        abNorm(A, a, b)

    Compute the elementwise matrix norm
    `` \\| A \\|_{b, a} := \\| (
        \\| (A_{1, :}) \\|_b, \\dots, \\| A_{n, :} \\|_b ) \\|_a ``.
    """
    function abNorm(A, a::Real, b::Real)
        # rowwise application of b-norm
        return norm(vec(mapslices(x -> norm(x, b), A, dims=[2])), a)
    end


    """
        matProj_2inf(A, gamma)

    Projects a matrix ``A`` to the ``2 \\to \\infty`` ball by decomposing the
    problem to ``n`` independent problems, one per row.
    """
    function matProj_2inf(A, gamma)
        return mapslices(x -> gamma * x / norm(x), A, dims=[2])
    end


    """
        dist_subg(v)

    Returns a subgradient for the distance function expressed as a vector.
    """
    function dist_subg(v)
        if norm(v) == 0.0
            return 0
        else
            return v / norm(v)
        end
    end


    """
        subg_sq21Norm(X, Xk; G=nothing)

    Returns a subgradient for the squared ``\\| \\cdot \\|_{2,1}`` matrix norm.
    """
    function subg_sq21Norm(X, Xk; G=nothing)
        cVal = abNorm(X - Xk, 1, 2)
        if G == nothing
            G = fill(0.0, size(X))
        end
        G[:] = 2 * cVal * mapslices(dist_subg, X - Xk, dims=[2])
        return G
    end


	"""
		subg_sq2infNorm(X, Xk; G=nothing)

	Compute a subgradient for the squared ``2,infty`` norm of the difference
	``X - X_k``.
	"""
	function subg_sq2infNorm(X, Xk; G=nothing)
		# maximizing indices
		indMax = argmax(vec(mapslices(norm, X - Xk, dims=[2])))
		if G == nothing
			G = fill(0.0, size(X))
		else
			fill!(G, 0.0)  # reset to zero
		end
		G[indMax, :] = 2 * ((X - Xk)[indMax, :])
		return G
	end


    """
        subDftMat(A, d, r)

    Return the first ``r`` columns from ``d`` randomly chosen rows of a
    ``2d \\times 2d`` DFT matrix.
    """
    function subDftMat(A, d, r)
        fmat = shuffle(fft(Matrix{Float64}(I, 2 * d, 2 * d), 2))
        return fmat[1:d, 1:r]
    end


    """
        genIncoherentMatrix(d, r)

    Generates a ``d \\times r`` matrix that satisfies
    `` \\| A \\|_{2,\\infty} \\leq \\sqrt{c r / d} \\| A \\|_{op}``.
    """
    function genIncoherentMatrix(d, r)
        return randn(d, r)   # fix this
    end


    """
        genSparseMatrix(d, r, corr_lvl)

    Generates an arbitrary sparse corruption matrix of size ``d \\times r``
    with `corr_lvl` fraction of nonzero entries.
    """
    function genSparseMatrix(d, r, corr_lvl)
        bernoulli(p) = trunc(Int, rand() <= p);
        # corruption matrix
        indMat = map(x -> bernoulli(corr_lvl), ones(d, r))
        S = 2 * randn(d, r)  # sparse corruption
        return indMat .* S
    end
end
