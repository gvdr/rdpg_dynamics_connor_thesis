using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
      OptimizationOptimisers, Random, Plots

using JSON
data = JSON.parsefile("./data/1_community_oscillation.json")

data["L_series"][1]
L_data = [[L[1] L[2]] for L in data["L_series"]] 

u = hcat([reshape(L[:, :]', 10,1) for L in L_data]...)



rng = Xoshiro(0)
u0 = u[:,1]
datasize = 15
tspan = (0.0f0, 14.0f0)
tsteps = range(tspan[1], tspan[2]; length = datasize)

ode_data = Array(u[:,1:15])

dudt2 = Chain(x -> x, Dense(10, 100, tanh), Dense(100, 50, tanh), Dense(50, 50, tanh), Dense(50, 10))
p, st = Lux.setup(rng, dudt2)
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(); saveat = tsteps)

function predict_neuralode(p)
    Array(prob_neuralode(u0, p, st)[1])
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss
end

# Do not plot by default for the documentation
# Users should change doplot=true to see the plots callbacks
function callback(state, l; doplot = false)
    println(l)
    # plot current prediction against data
    if doplot
        pred = predict_neuralode(state.u)
        plt = scatter(tsteps, ode_data[2, :]; label = "data")
        scatter!(plt, tsteps, pred[2, :]; label = "prediction")
        display(plot(plt))
    end
    return false
end

pinit = ComponentArray(p)
callback((; u = pinit), loss_neuralode(pinit); doplot = true)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

result_neuralode = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.05); callback = callback, maxiters = 300)

optprob2 = remake(optprob; u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(
    optprob2, Optim.BFGS(; initial_stepnorm = 0.01); callback, allow_f_increases = false, maxiters = 100)

callback((; u = result_neuralode2.u), loss_neuralode(result_neuralode2.u); doplot = true)

using Serialization
serialize("./models/1_community_oscillation/25-07-2025-L.jls", result_neuralode2.u)

