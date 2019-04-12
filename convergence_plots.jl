using ArgParse
using CSV
using DataFrames
using LinearAlgebra
using Printf
using Random
using Statistics

include("src/CompOpt.jl")


zero_pad!(v, lenTotal) = append!(v, zeros(lenTotal - length(v)))

#= bilinear sensing =#
function bilin_experiment(d1, d2, iList, r, iters, delta, noise_lvl;
	                      algo=:subgrad)
	df = DataFrame(k=collect(1:iters))
	for i in iList
		m = i * r * (d1 + d2)
		prob = CompOpt.genBilinProb(d1, d2, m, r, noise_lvl)
		if algo == :subgrad
			_, _, ds = CompOpt.pSgd_init(prob, iters, delta, eps=1e-12)
		else
			_, _, ds = CompOpt.bilinProxlin_init(prob, delta, iters, eps=1e-12)
		end
		zero_pad!(ds, iters); df[Symbol("err_$(i)")] = ds
	end
	return df
end


#= symmetrized quadratic sensing =#
function quad_experiment(d, iList, r, iters, delta, noise_lvl; problem=:symmetrized,
	                     algo=:subgrad)
	df = DataFrame(k=collect(1:iters))
	for i in iList
		if problem == :symmetrized
			pGen = CompOpt.genSymQuadProb; m = i * r * d
		else
			pGen = CompOpt.genQuadProb; m = i * (r^2) * d
		end
		prob = pGen(d, m, r, noise_lvl)
		if algo == :subgrad
			_, ds = CompOpt.pSgd_init(prob, iters, delta, eps=1e-12)
		else
			_, ds = CompOpt.symQuadProxlin_init(prob, delta, iters, eps=1e-12)
		end
		zero_pad!(ds, iters); df[Symbol("err_$(i)")] = ds
	end
	return df
end


#= matrix completion =#
function matcomp_experiment(d, r, iters, delta; algo=:subgrad)
	df = DataFrame(k=collect(1:iters))
	for sample_freq in [0.05; 0.1; 0.15; 0.2]
		prob = CompOpt.genMatCompProb(d, r, sample_freq)
		if algo == :subgrad
			_, ds = CompOpt.pSgd_init(prob, iters, delta)
		else
			_, ds = CompOpt.matCompProxLinear_init(prob, iters, delta)
		end
		zero_pad!(ds, iters); df[Symbol("err_$(sample_freq)")] = ds
	end
	return df
end


function rpca_experiment(d, r, iters, delta; algo=:subgrad)
	df = DataFrame(k=collect(1:iters))
	for corr_lvl in [0.0; 0.1; 0.2; 0.3; 0.4]
		prob = CompOpt.genRpcaProb(d, r, corr_lvl)
		if algo == :subgrad
			_, ds = CompOpt.pSgd_init(prob, iters, delta, mode=:noneuclidean)
		else
			_, ds = CompOpt.rpcaProxLin_init(prob, iters, delta, inner_eps=1e-5,
			                                 maxIt=750)
		end
		zero_pad!(ds, iters); df[Symbol("err_$(corr_lvl)_$(algo)")] = ds
	end
	return df
end


function main()
	# parse arguments
	s = ArgParseSettings(description="""
						 Generates a set of synthetic problem instances for
						 a given ratio m / dim and failure probabilities and
						 solves them using a specified method.
						 Outputs the convergence history in a .csv file.""")
	@add_arg_table s begin
		"--d1"
			help = "Dimension 1 of the problem"
			arg_type = Int
			default = 200
		"--d2"
			help = "Dimension 2 of the problem (only for bilinear sensing)"
			arg_type = Int
			default = 200
		"--r"
			help = "The rank of the matrices involved"
			arg_type = Int
			default = 5
		"--seed"
			help = "The seed of the RNG"
			arg_type = Int
			default = 999
		"--corr_lvl"
			help = """
			The level of corruption, which translates to fraction of corrupted
			measurements in matrix sensing and robust pca."""
			arg_type = Float64
			default = 0.25
		"--prob_type"
			help =
				"""
				The type of the problem. `quadratic` results in a quadratic problem,
				`sym_quadratic` in a quadratic problem with symmetrized measurements,
				and `bilinear` results in a bilinear problem.
				`matcomp` results in a matrix completion problem, while `rpca`
				results in an instance of robust PCA."""
			range_tester = (x -> lowercase(x) in [
				"quadratic", "sym_quadratic", "bilinear", "matcomp", "rpca"])
			default = "bilinear"
		"--algo_type"
			help =
				"""
				The iterative algorithm to be used. `subgradient` denotes the
				subgradient method with geometrically decaying step size or
				the Polyak step size when the minimum value is known. `proxlinear`
				denotes the prox-linear method, tailored to the specific problem
				at hand."""
			range_tester = (x -> lowercase(x) in [
				"subgradient", "proxlinear"])
			default = "subgradient"
		"--iters"
			help = "The number of iterations for minimization"
			arg_type = Int
			default = 1000
		"--i"
			help = "The ratio of measurements to the problem's dimension"
			arg_type = Int
			nargs = '*'
		"--delta"
			help = "The initial distance to ground truth"
			arg_type = Float64
			default = 0.95
	end
	parsed = parse_args(s)
	d1, d2, r, rnd_seed = parsed["d1"], parsed["d2"], parsed["r"], parsed["seed"]
	prob_type, iters, delta = parsed["prob_type"], parsed["iters"], parsed["delta"]
	corr_lvl = parsed["corr_lvl"]
	algo_type = parsed["algo_type"]
	iList = parsed["i"]
	# seed RNG
	Random.seed!(rnd_seed)
	df = nothing;
	if prob_type == "quadratic"
		df = quad_experiment(d1, iList, r, iters, delta, corr_lvl, problem=:quadratic)
	elseif prob_type == "sym_quadratic"
		algo = (algo_type == "subgradient") ? :subgrad : :proxlin;
		df = quad_experiment(d1, iList, r, iters, delta, corr_lvl, problem=:symmetrized,
		                     algo=algo)
	elseif prob_type == "bilinear"
		algo = (algo_type == "subgradient") ? :subgrad : :proxlin;
		df = bilin_experiment(d1, d2, iList, r, iters, delta, corr_lvl,
		                      algo=algo)
	elseif prob_type == "matcomp"
		algo = (algo_type == "subgradient") ? :subgrad : :proxlin;
		df = matcomp_experiment(d1, r, iters, delta, algo=algo)
	else
		algo = (algo_type == "subgradient") ? :subgrad : :proxlin;
		df = rpca_experiment(d1, r, iters, delta, algo=algo)
	end
	if (prob_type == "bilinear" || prob_type == "sym_quadratic" ||
	    prob_type == "quadratic")
		fname = "err_$(prob_type)_$(r)_$(algo_type)_$(corr_lvl).csv"
	else
		fname = "err_$(prob_type)_$(r)_$(algo_type).csv"
	end
	CSV.write(fname, df)
end

main()
