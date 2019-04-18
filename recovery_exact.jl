using ArgParse
using LinearAlgebra
using Printf
using PyPlot
using Random
using Statistics

include("src/CompOpt.jl")

LBLUE = "#908cc0"
MBLUE = "#519cc8"
HBLUE = "#1d5996"
LRED = "#cb5501"
MRED = "#f1885b"
HRED = "#b3001e"

zero_pad!(v, lenTotal) = append!(v, zeros(lenTotal - length(v)))

#= bilinear sensing =#
function bilin_experiment(d1, d2, r, iters, delta; algo_type=:subgradient)
	r = 5; m = 8 * r * (d1 + d2)
	prob_mild = CompOpt.genBilinProb(d1, d2, m, r, 0.25)
	prob_high = CompOpt.genBilinProb(d1, d2, m, r, 0.40)
	ds_mild = nothing; ds_high = nothing
	style = (algo_type == :subgradient) ? "-" : ".-"  # line style
	println("Running for rank $(r)...")
	if algo_type == :subgradient
		_, _, ds_mild = CompOpt.pSgd_init(prob_mild, iters, delta)
		_, _, ds_high = CompOpt.pSgd_init(prob_high, iters, delta)
	else
		_, _, ds_mild = CompOpt.proxlin_init(prob_mild, delta, iters)
		_, _, ds_high = CompOpt.proxlin_init(prob_high, delta, iters)
	end
	semilogy(collect(1:length(ds_mild)), ds_mild, color=LBLUE, style,
	         label=L"$ (r, p) = (5, 0.25) $")
	semilogy(collect(1:length(ds_high)), ds_high, color=MBLUE, style,
	         label=L"$ (r, p) = (5, 0.40) $")
	r = 10; m = 8 * r * (d1 + d2)
	prob_mild = CompOpt.genBilinProb(d1, d2, m, r, 0.25)
	prob_high = CompOpt.genBilinProb(d1, d2, m, r, 0.40)
	println("Running for rank $(r)...")
	if algo_type == :subgradient
		_, _, ds_mild = CompOpt.pSgd_init(prob_mild, iters, delta)
		_, _, ds_high = CompOpt.pSgd_init(prob_high, iters, delta)
	else
		_, _, ds_mild = CompOpt.proxlin_init(prob_mild, delta, iters)
		_, _, ds_high = CompOpt.proxlin_init(prob_high, delta, iters)
	end
	semilogy(collect(1:length(ds_mild)), ds_mild, color=HBLUE, style,
	         label=L"$ (r, p) = (10, 0.25) $")
	semilogy(collect(1:length(ds_high)), ds_high, color="black", style,
	         label=L"$ (r, p) = (10, 0.40) $")
	xlabel(L"$ k $"); ylabel("Normalized error")
	title("Bilinear sensing - $(algo_type) method"); legend(); show()
	# compare with gradient descent
	println("Comparing both with gradient descent...")
	prob_n5 = CompOpt.genBilinProb(d1, d2, 8 * 5 * (d1 + d2), 5)
	prob_n10 = CompOpt.genBilinProb(d1, d2, 8 * 10 * (d1 + d2), 10)
	_, _, ds_n5 = CompOpt.pSgd_init(prob_n5, iters, delta)
	_, _, ds_n10 = CompOpt.pSgd_init(prob_n10, iters, delta)
	_, _, ds_grad5 = CompOpt.bilinNaiveGD_init(prob_n5, delta, iters, 0.001)
	_, _, ds_grad10 = CompOpt.bilinNaiveGD_init(prob_n10, delta, iters, 0.001)
	figure();
	semilogy(collect(1:length(ds_n5)), ds_n5, color=LBLUE,
	         label=L"$ r = 5 $")
	semilogy(collect(1:length(ds_n10)), ds_n10, color=HBLUE,
	         label=L"$ r = 10 $")
	semilogy(collect(1:length(ds_grad5)), ds_grad5, color=LBLUE, "--",
	         label=L"$ r = 5 $ - grad")
	semilogy(collect(1:length(ds_grad10)), ds_grad10, color=HBLUE, "--",
	         label=L"$ r = 10 $ - grad")
	xlabel(L"$ k $"); ylabel("Normalized error")
	title("Bilinear sensing - Polyak subgradient vs. gradient descent")
	legend(); show()
end


#= symmetrized quadratic sensing =#
function quad_experiment(d, r, iters, delta; algo_type=:subgradient)
	r = 5; m = 8 * r * d
	prob_mild = CompOpt.genSymQuadProb(d, m, r, 0.25)
	prob_high = CompOpt.genSymQuadProb(d, m, r, 0.40)
	ds_mild = nothing; ds_high = nothing
	style = (algo_type == :subgradient) ? "-" : ".-"  # line style
	println("Running for rank $(r)...")
	if algo_type == :subgradient
		_, ds_mild = CompOpt.pSgd_init(prob_mild, iters, delta, eps=1e-12)
		_, ds_high = CompOpt.pSgd_init(prob_high, iters, delta, eps=1e-12)
	else
		_, ds_mild = CompOpt.proxlin_init(prob_mild, delta, iters, eps=1e-12)
		_, ds_high = CompOpt.proxlin_init(prob_high, delta, iters, eps=1e-12)
	end
	semilogy(collect(1:length(ds_mild)), ds_mild, color=LBLUE, style,
	         label=L"$ (r, p) = (5, 0.25) $")
	semilogy(collect(1:length(ds_high)), ds_high, color=MBLUE, style,
	         label=L"$ (r, p) = (5, 0.40) $")
	r = 10; m = 8 * r * d
	prob_mild = CompOpt.genSymQuadProb(d, m, r, 0.25)
	prob_high = CompOpt.genSymQuadProb(d, m, r, 0.40)
	println("Running for rank $(r)...")
	if algo_type == :subgradient
		_, ds_mild = CompOpt.pSgd_init(prob_mild, iters, delta, eps=1e-12)
		_, ds_high = CompOpt.pSgd_init(prob_high, iters, delta, eps=1e-12)
	else
		_, ds_mild = CompOpt.proxlin_init(prob_mild, delta, iters, eps=1e-12)
		_, ds_high = CompOpt.proxlin_init(prob_high, delta, iters, eps=1e-12)
	end
	semilogy(collect(1:length(ds_mild)), ds_mild, color=HBLUE, style,
	         label=L"$ (r, p) = (10, 0.25) $")
	semilogy(collect(1:length(ds_high)), ds_high, color="black", style,
	         label=L"$ (r, p) = (10, 0.40) $")
	xlabel(L"$ k $"); ylabel("Normalized error")
	title("Quadratic sensing - $(algo_type) method"); legend(); show()
	# compare with gradient descent
	println("Comparing both with gradient descent...")
	prob_n5 = CompOpt.genSymQuadProb(d, 8 * 5 * d, 5)
	prob_n10 = CompOpt.genSymQuadProb(d, 8 * 10 * d, 10)
	_, ds_n5 = CompOpt.pSgd_init(prob_n5, iters, delta)
	_, ds_n10 = CompOpt.pSgd_init(prob_n10, iters, delta)
	_, ds_grad5 = CompOpt.symQuadNaiveGD_init(prob_n5, delta, iters, 0.0001)
	_, ds_grad10 = CompOpt.symQuadNaiveGD_init(prob_n10, delta, iters, 0.0001)
	figure()
	semilogy(collect(1:length(ds_n5)), ds_n5, color=LBLUE,
	         label=L"$ r = 5 $")
	semilogy(collect(1:length(ds_n10)), ds_n10, color=HBLUE,
	         label=L"$ r = 10 $")
	semilogy(collect(1:length(ds_grad5)), ds_grad5, color=LBLUE, "--",
	         label=L"$ r = 5 $ - grad")
	semilogy(collect(1:length(ds_grad10)), ds_grad10, color=HBLUE, "--",
	         label=L"$ r = 10 $ - grad")
	xlabel(L"$ k $"); ylabel("Normalized error")
	title("Quadratic sensing - Polyak subgradient vs. gradient descent")
	legend(); show()
end


#= matrix completion =#
function matcomp_experiment(d, r, iters, delta; algo_type=:subgradient)
	pCol = [LBLUE, HBLUE, "black"]; idx = 0
	for sample_freq in 0.1:0.05:0.2
		println("Running for sample frequency $(sample_freq)")
		idx += 1
		prob = CompOpt.genMatCompProb(d, r, sample_freq)
		if algo_type == :subgradient
			_, ds = CompOpt.pSgd_init(prob, iters, delta)
			# compare with gd
			_, ds_grad = CompOpt.matCompNaiveGD_init(prob, delta, iters, 0.004)
			zero_pad!(ds, iters); zero_pad!(ds_grad, iters)
			semilogy(collect(1:iters), ds, color=pCol[idx],
			         label=latexstring("p = $sample_freq"));
			semilogy(collect(1:iters), ds_grad, color=pCol[idx], "--")
		else
			_, ds = CompOpt.matCompProxLinear_init(prob, iters, delta)
			zero_pad!(ds, iters)
			semilogy(collect(1:iters), ds, color=pCol[idx], ".-",
			         label=latexstring("p = $sample_freq"))
		end
	end
	xlabel(L"$ k $"); ylabel("Normalized error")
	title("Matrix completion - $(algo_type) method vs. gradient descent")
	legend(); show()
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
				The type of the problem. `quadratic` results in a quadratic
				problem with symmetrized measurements, and `bilinear` results
				in a bilinear problem.
				`matcomp` results in a matrix completion problem, while `rpca`
				results in an instance of robust PCA."""
			range_tester = (x -> lowercase(x) in [
				"quadratic", "bilinear", "matcomp", "rpca"])
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
		"--delta"
			help = "The initial distance to ground truth"
			arg_type = Float64
			default = 0.95
	end
	parsed = parse_args(s)
	d1, d2, r, rnd_seed = parsed["d1"], parsed["d2"], parsed["r"], parsed["seed"]
	prob_type, iters, delta = parsed["prob_type"], parsed["iters"], parsed["delta"]
	algo_type = parsed["algo_type"]
	# seed RNG
	Random.seed!(rnd_seed)
	df = nothing;
	if prob_type == "quadratic"
		algo = (algo_type == "subgradient") ? :subgradient : :proxlinear
		quad_experiment(d1, r, iters, delta, algo_type=algo)
	elseif prob_type == "bilinear"
		algo = (algo_type == "subgradient") ? :subgradient : :proxlinear
		bilin_experiment(d1, d2, r, iters, delta, algo_type=algo)
	elseif prob_type == "matcomp"
		algo = (algo_type == "subgradient") ? :subgradient : :proxlinear
		df = matcomp_experiment(d1, r, iters, delta, algo_type=algo)
	else
		throw(Exception("Not implemented yet"))
	end
end

main()
