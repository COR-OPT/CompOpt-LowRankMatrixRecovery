using ArgParse
using DataFrames
using LinearAlgebra
using Printf
using PyPlot
using Random
using Statistics

include("src/CompOpt.jl")


zero_pad!(v, lenTotal) = append!(v, zeros(lenTotal - length(v)))

#= symmetrized quadratic sensing =#
function quad_dense(d, i, delta, dLvlList)
	r = 5
	for dLvl in dLvlList
		figure()
		# additive gaussian noise
		prob = CompOpt.genSymQuadProb(d, i * r * d, r, 0.25, denseLvl=dLvl)
		_, ds_sg = CompOpt.pSgd_init(prob, 500, delta, eps=1e-12)
		_, ds_pl = CompOpt.proxlin_init(prob, 20, delta, eps=1e-12)
		subplot(211); semilogy(collect(1:length(ds_sg)), ds_sg)
		xlabel(L"$ k $"); ylabel("Normalized error")
		title("Subgradient method")
		subplot(212); semilogy(collect(1:length(ds_pl)), ds_pl)
		xlabel(L"$ k $"); ylabel("Normalized error")
		title("Prox-linear method")
		show()
	end
end


#= matrix completion =#
function matcomp_dense(d, delta, dLvlList)
	r = 8
	for dLvl in dLvlList
		figure()
		prob = CompOpt.genMatCompProb(d, r, 0.25, denseLvl=dLvl)
		_, ds_sg = CompOpt.pSgd_init(prob, 500, delta, eps=1e-12)
		_, ds_pl = CompOpt.proxlin_init(prob, 20, delta)
		subplot(211); semilogy(collect(1:length(ds_sg)), ds_sg)
		xlabel(L"$k$"); ylabel(L"$ \\| X_k X_k^\\top - M_{\\sharp} \\|_F / \\| M_{\\sharp} \\|_F")
		title("Subgradient method")
		subplot(212); semilogy(collect(1:length(ds_pl)), ds_pl)
		xlabel(L"$k$"); ylabel(L"$ \\| X_k X_k^\\top - M_{\\sharp} \\|_F / \\| M_{\\sharp} \\|_F")
		title("Prox-linear method")
		show()
	end
end


function main()
	# parse arguments
	s = ArgParseSettings(description="""
						 Solves symmetrized quadratic sensing and matrix
						 completion problems with dense additive noise, plotting
						 the convergence history.""")
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
		"--i"
			help = "The ratio of measurements to the problem's dimension"
			arg_type = Int
			default = 8
		"--delta"
			help = "The initial distance to ground truth"
			arg_type = Float64
			default = 0.8
	end
	parsed = parse_args(s)
	d, rnd_seed, i = parsed["d"], parsed["seed"], parsed["i"]
	delta = parsed["delta"]; dLvlList = parsed["denseLvl"]; i = parsed["i"]
	# seed RNG
	Random.seed!(rnd_seed)
	println("Running quadratic experiment...")
	quad_dense(d, i, delta, dLvlList)
	println("Running matcomp experiment...")
	matcomp_dense(d, delta, dLvlList)
end

main()
