"""
A module implementing nonsmooth optimization methods for a variety of
statistical problems that exhibit sharpness and weak convexity.
"""
module LinearConv

    include("Utils.jl")

    using LinearAlgebra
    using Random
    using Statistics
    using Arpack  # required for eigs


    struct QuadProb
        y :: Array{Float64, 1}
        X :: Array{Float64, 2}
        A :: Array{Float64, 2}
        pfail :: Float64
    end


    struct BilinProb
        y :: Array{Float64, 1}
        X :: Array{Float64, 2}
        A :: Array{Float64, 2}
        B :: Array{Float64, 2}
        pfail :: Float64
    end


    """
        genQuadProb(d, n, m, r, noise_lvl=0.0)

    Generates a quadratic sensing problem in dimensions ``d \\times n`` where
    ``\\rank(X) = r`` with a desired noise level.
    """
    function genQuadProb(d, n, m, r, noise_lvl=0.0)
        A = randn(m, d); X = Utils.genMtx(d, n, r)
        y = mapslices(norm, A * X, dims=[2])  # get measurements
        Utils.corrupt_measurements!(y, noise_lvl, :gaussian)
        return QuadProb(y, X, A, noise_lvl)
    end


    """
        genBilinProb(d1, d2, m, r, noise_lvl=0.0)

    Generates a bilinear sensing problem in dimensions ``d_1 \\times d_2``
    where ``\\rank(X) = r`` with a desired noise level.
    """
    function genBilinProb(d1, d2, m, r, noise_lvl=0.0)
        A = randn(m, d1); B = randn(m, d2); X = Utils.genMtx(d1, d2, r)
        # factorize S
        F = svd(X); Xs = F.S[1:r]; XU = F.U[1:end, 1:r] XV = F.V[1:end, 1:r]
        # sum all factors
        y = sum(Xs[i] * (A * XU[:, i]) .* (B * XV[:, i]) for i in 1:r)
        Utils.corrupt_measurements!(y, noise_lvl, :gaussian)
        return BilinProb(y, X, A, B, pfail)
    end


    """
        quadRes(qProb, Xcurr)

    Compute the residual of the quadratic sensing model at the current
    estimate `Xcurr`.
    """
    function quadRes(qProb, Xcurr)
        r = mapslices(norm, qProb.A * Xcurr, dims=[2])
        broadcast!(-, r, r, qProb.y)
        return r
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
        bilinRes(bProb, Xcurr)

    Compute the residual of the bilinear sensing model at the current estimate
    `Xcurr`.
    """
    function bilinRes(bProb, Xcurr)
        R = bProb.A * Xcurr; m = length(bProb.y)
        r = [R[i, :]' * B[i, :] for i = 1:m];
        return r
    end


    """
        bilinRobustLoss(bProb, Xcurr)

    Compute the robust loss for the bilinear sensing problem given a problem
    instance `bProb` and the current estimate `Xcurr`.
    """
    function bilinRobustLoss(bProb, Xcurr)
        r = bilinRes(bProb, Xcurr)
        return (1 / length(bProb.y)) * norm(r, 1)
    end


    """
        quadSubgrad!(qProb, Xcurr; grad=nothing)

    Compute the subgradient at `Xcurr` for the quadratic sensing problem.
    """
    function quadSubgrad!(qProb, Xcurr; grad=nothing)
        m = length(qProb.y); d, n = size(Xcurr)
        if grad == nothing
            grad = fill(0.0, (d, n))
        end
        # sign and A * X
        rSign = map.(sign, quadRes(qProb, Xcurr)); R = qProb.A * Xcurr
        # compute subgradient
        grad[:] = (1 / m) * qProb.A' * (rSign .* R)
        return grad
    end


    """
        bilinSubgrad!(bProb, Xcurr; grad=nothing)

    Compute the subgradient at `Xcurr` for the bilinear sensing problem.
    """
    function bilinSubgrad!(bProb, Xcurr; grad=nothing)
        m = length(bProb.y); d1, d2 = size(Xcurr)
        if grad == nothing
            grad = fill(0.0, (d, n))
        end
        rSign = map.(sign, bilinRes(bProb, Xcurr))
        grad[:] = (1 / m) * bProb.A' * (rSign .* bProb.B)
        return grad
    end


    """
        pSgd(qProb::QuadProb, Xinit, iters; λ = 1.0, rho = 0.98)

    Apply the projected subgradient method with geometrically decaying step
    to the quadratic sensing problem for `iters` iterations, given a problem
    instance `qProb` and an initial estimate `Xinit`.
    """
    function pSgd(qProb::QuadProb, Xinit, iters; λ = 1.0, rho = 0.98)
        Xtrue = qProb.X; d, n = size(Xtrue); grad = fill(0.0, (d, n))
        q = λ; dist = fill(0.0, iters);
        for i = 1:iters
            quadSubgrad!(qProb, Xinit, grad=grad)
            broadcast!(-, Xinit, Xinit, q * grad / norm(grad))
            q *= rho; dist[i] = norm(Xinit - Xtrue)
        end
        return Xinit, dist
    end


    """
        pSgd(bProb::BilinProb, Xinit, iters; λ = 1.0, rho = 0.98)

    Apply the projected subgradient method with geometrically decaying step
    to the bilinear sensing problem for `iters` iterations, given a problem
    instance `qProb` and an initial estimate `Xinit`.
    """
    function pSgd(bProb::BilinProb, Xinit, iters; λ = 1.0, rho = 0.98)
        Xtrue = bProb.X; d, n = size(Xtrue); grad = fill(0.0, (d, n))
        q = λ; dist = fill(0.0, iters);
        for i = 1:iters
            bilinSubgrad!(bProb, Xinit, grad=grad)
            broadcast!(-, Xinit, Xinit, q * grad / norm(grad))
            q *= rho; dist[i] = norm(Xinit - Xtrue)
        end
        return Xinit, dist
    end
end
