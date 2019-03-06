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
