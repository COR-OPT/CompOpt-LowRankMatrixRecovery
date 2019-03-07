module Utils

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
end
