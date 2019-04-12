using ArgParse
using CSV
using DataFrames
using LinearAlgebra
using Printf
using Random
using Statistics

include("src/CompOpt.jl")


zero_pad!(v, lenTotal) = append!(v, zeros(lenTotal - length(v)))

#= symmetrized quadratic sensing =#
function quad_experiment(d, i, delta, dLvlList)
	df_sg = DataFrame(k=collect(1:500)); df_pl = DataFrame(k=collect(1:20))
	r = 5
	for dLvl in dLvlList
		# additive gaussian noise
		prob = CompOpt.genSymQuadProb(d, i * r * d, r, 0.25, denseLvl=dLvl)
		_, ds_sg = CompOpt.pSgd_init(prob, 500, delta, eps=1e-12)
		_, ds_pl = CompOpt.symQuadProxlin_init(prob, delta, 20, eps=1e-12)
		zero_pad!(ds_sg, 500); zero_pad!(ds_pl, 20)
		df_sg[Symbol("err_$(dLvl)_$(r)")] = ds_sg
		df_pl[Symbol("err_$(dLvl)_$(r)")] = ds_pl
	end
	return df_sg, df_pl
end


#= matrix completion =#
function matcomp_experiment(d, delta, dLvlList)
	df_sg = DataFrame(k=collect(1:500)); df_pl = DataFrame(k=collect(1:20))
	r = 8
	for dLvl in dLvlList
		prob = CompOpt.genMatCompProb(d, r, 0.25, denseLvl=dLvl)
		_, ds_sg = CompOpt.pSgd_init(prob, 500, delta, eps=1e-12)
		_, ds_pl = CompOpt.matCompProxLinear_init(prob, 20, delta)
		zero_pad!(ds_sg, 500); zero_pad!(ds_pl, 20)
		df_sg[Symbol("err_$(dLvl)")] = ds_sg
		df_pl[Symbol("err_$(dLvl)")] = ds_pl
	end
	return df_sg, df_pl
end


function main()
	# parse arguments
	s = ArgParseSettings(description="""
						 Generates a set of synthetic problem instances for
						 a given ratio m / dim and failure probabilities and
						 solves them using a specified method.
						 Outputs the convergence history in a .csv file.""")
	@add_arg_table s begin
		"--d"
			help = "Dimension of the problem"
			arg_type = Int
			default = 200
		"--seed"
			help = "The seed of the RNG"
			arg_type = Int
			default = 999
		"--denseLvl"
			help = """
			The levels of dense corruption, which translate to intensity of
			additive Gaussian noise"""
			arg_type = Float64
			nargs = '*'
		"--iters"
			help = "The number of iterations for minimization"
			arg_type = Int
			default = 1000
		"--i"
			help = "The ratio of measurements to the problem's dimension"
			arg_type = Int
			default = 8
		"--delta"
			help = "The initial distance to ground truth"
			arg_type = Float64
			default = 1.0
	end
	parsed = parse_args(s)
	d, rnd_seed, i = parsed["d"], parsed["seed"], parsed["i"]
	delta = parsed["delta"]; dLvlList = parsed["denseLvl"]; i = parsed["i"]
	# seed RNG
	Random.seed!(rnd_seed)
	println("Running quadratic experiment...")
	df_sg, df_pl = quad_experiment(d, i, delta, dLvlList)
	CSV.write("err_symquad_dense_$(d)_subgrad.csv", df_sg)
	CSV.write("err_symquad_dense_$(d)_proxlin.csv", df_pl)
	println("Running matcomp experiment...")
	df_sg, df_pl = matcomp_experiment(d, delta, dLvlList)
	CSV.write("err_matcomp_dense_$(d)_subgrad.csv", df_sg)
	CSV.write("err_matcomp_dense_$(d)_proxlin.csv", df_pl)
end

main()
