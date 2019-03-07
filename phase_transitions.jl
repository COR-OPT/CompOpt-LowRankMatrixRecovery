using ArgParse
using LinearAlgebra
using Printf
using Random
using Statistics

include("src/LinearConv.jl")


#= setup the bilinear sensing experiment =#
function bilin_experiment(d1, d2, i, r, iters, delta, reps;
						  success_eps=1e-5)
	for noise_lvl = 0:0.02:0.48
		success = 0; m = i * r * (d1 + d2)
		for k = 1:reps
			prob = LinearConv.genBilinProb(d1, d2, m, r, noise_lvl)
			_, _, ds = LinearConv.pSgd_init(prob, iters, delta, eps=success_eps)
			success += (ds[end] <= success_eps) ? 1 : 0
		end
		@printf("%d, %.2f, %.2f\n", i, noise_lvl, success / reps)
	end
end


#= setup the quadratic sensing experiment =#
function quad_experiment(d, i, r, iters, delta, reps;
						 success_eps=1e-5, problem=:symmetrized)
	if problem == :symmetrized
		pGen = LinearConv.genSymQuadProb; m = i * r * d
	else
		pGen = LinearConv.genQuadProb; m = i * (r^2) * d
	end
	for noise_lvl = 0:0.02:0.48
		success = 0
		for k = 1:reps
			prob = pGen(d, m, r, noise_lvl)
			_, ds = LinearConv.pSgd_init(prob, iters, delta, eps=success_eps)
			success += (ds[end] <= success_eps) ? 1 : 0
		end
		@printf("%d, %.2f, %.2f\n", i, noise_lvl, success / reps)
	end
end


function main()
	# parse arguments
	s = ArgParseSettings(description="""
						 Generates a set of synthetic problem instances for
						 a given ratio m / dim and a range of failure
						 probabilities, solves them using the subgradient method, and
						 outputs the percentage of successful recoveries.""")
	@add_arg_table s begin
		"--d1"
			help = "Dimension 1 of the problem"
			arg_type = Int
			default = 100
		"--d2"
			help = "Dimension 2 of the problem"
			arg_type = Int
			default = 100
		"--r"
			help = "The rank of the matrices involved"
			arg_type = Int
			default = 5
		"--seed"
			help = "The seed of the RNG"
			arg_type = Int
			default = 999
		"--prob_type"
			help =
				"""
				The type of the problem. `quadratic` results in a quadratic problem,
				`sym_quadratic` in a quadratic problem with symmetrized measurements,
				and `bilinear` results in a bilinear problem."""
			range_tester = (x -> lowercase(x) in [
				"quadratic", "sym_quadratic", "bilinear"])
			default = "bilinear"
		"--iters"
			help = "The number of iterations for minimization"
			arg_type = Int
			default = 1000
		"--repeats"
			help = "The number of repeats for generating success rates"
			arg_type = Int
			default = 50
		"--success_dist"
			help = """The desired reconstruction distance. Iterates whose
				   normalized distance is below this threshold are considered
				   exact recoveries."""
			arg_type = Float64
			default = 1e-5
		"--i"
			help = "The ratio of measurements to the problem's dimension"
			arg_type = Int
			default = 1
		"--delta"
			help = "The initial distance to ground truth"
			arg_type = Float64
			default = 0.95
	end
	parsed = parse_args(s)
	d1, d2, r, rnd_seed = parsed["d1"], parsed["d2"], parsed["r"], parsed["seed"]
	prob_type, iters, delta = parsed["prob_type"], parsed["iters"], parsed["delta"]
	sdist, repeats = parsed["success_dist"], parsed["repeats"]
	i = parsed["i"]
	# seed RNG
	Random.seed!(rnd_seed)
	if prob_type == "quadratic"
		quad_experiment(d1, i, r, iters, delta, repeats, success_eps=sdist,
						problem=:quadratic)
	elseif prob_type == "sym_quadratic"
		quad_experiment(d1, i, r, iters, delta, repeats, success_eps=sdist,
						problem=:symmetrized)
	else
		bilin_experiment(d1, d2, i, r, iters, delta, repeats, success_eps=sdist)
	end
end

main()
