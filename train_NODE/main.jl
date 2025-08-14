using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL, OptimizationOptimisers, Random, Plots
using JSON
using MLUtils
using Lux, Optimization, OptimizationOptimisers, OrdinaryDiffEq, SciMLSensitivity, MLUtils,
      Random, ComponentArrays
using Optim

data = JSON.parsefile("./data/2_communities_joining.json")

n = 11 # Total Number of Nodes in network
d = 2 # Number of singular values taken, for this paper will leave at 2


L_data = [[L[1] L[2]] for L in data["L_series"]] 
R_data = [[R[1] R[2]] for R in data["R_series"]] 

L_R_translation = []
for i in eachindex(L_data)
    if L_data[i] ≈ R_data[i]
        println("same")
        push!(L_R_translation, [1 0; 0 1])
    elseif L_data[i]*[1 0; 0 -1] ≈ R_data[i]
        println("alternate")
        push!(L_R_translation, [1 0; 0 -1])
    else 
        println("FAIL")
    end
end


u = Float32.(hcat([reshape(L', n*d,1) for L in L_data]...))


rng = Xoshiro(1254)
u0 = u[:,1] 
datasize = 15
tspan = (0.0f0, 14.0f0)
tsteps = range(tspan[1], tspan[2]; length = datasize)



ode_data = collect(Array.(eachcol(Array(u[:,1:datasize]))))
data_translation = L_R_translation[1:datasize]



chain = Chain(x -> x, Dense(n*d, 128, celu), Dense(128, 128, celu), Dense(128, 64, celu), Dense(64, n*d))
p, st = Lux.setup(rng, chain)
ps_ca = ComponentArray(p)

function dudt_(u, p, t)
    c = chain(u, p, st)[1]
    return c
end

# prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(); saveat = tsteps)
prob_neuralode = ODEProblem{false}(dudt_, u0, tspan, ps_ca)

function predict_neuralode(fullp, time_batch)
    A = Array(solve(prob_neuralode, Tsit5(), p = fullp, saveat = time_batch[2]))
    return A
end

function pred_to_array(pred, translation)
    array_pred = pred*(pred*translation)'

    return array_pred
end

function lower_than_zero_loss(array)
    loss = 0
    for m in 1:size(array, 1)
        for n in 1:size(array, 2)
            if array[m,n] < 0
                if m!=n
                    loss-=array[m,n]
                end
            end
        end
    end
    return loss
end

function greater_than_one_loss(array)
    loss = 0
    for m in 1:size(array, 1)
        for n in 1:size(array, 2)
            if array[m,n] > 1 
                if m!=n
                    loss+=array[m,n]-1 # Only loss for the distance over 1
                end
            end
        end
    end
    return loss
end

function loss_neuralode(p, data)
    batch, time_batch = data
    pred = predict_neuralode(p, time_batch)
    loss = sum(abs2, hcat(time_batch[1]...) .- pred)
    reshaped_preds = [reshape(col, n, d) for col in eachcol(pred)]
    pred_arrays = pred_to_array.(reshaped_preds, data_translation[Int.(collect(time_batch[2])).+1])
    less_than_zero_loss = sum(lower_than_zero_loss.(pred_arrays))
    more_than_one_loss = sum(greater_than_one_loss.(pred_arrays))
    return loss + less_than_zero_loss + more_than_one_loss
end

function predict_adjoint(fullp, time_batch)
    Array(solve(prob, Tsit5(), p = fullp, saveat = time_batch))
end

function loss_adjoint(fullp, data)
    batch, time_batch = data
    pred = predict_adjoint(fullp, time_batch)
    sum(abs2, batch .- pred)
end

# Do not plot by default for the documentation
# Users should change doplot=true to see the plots callbacks
function callback(state, l; doplot = false)
    println(l)
    # plot current prediction against data
    # if doplot
    #     pred = predict_neuralode(state.u)
    #     plt = scatter(tsteps, ode_data[2, :]; label = "data")
    #     scatter!(plt, tsteps, pred[2, :]; label = "prediction")
    #     display(plot(plt))
    # end
    return false
end


train_loader = MLUtils.DataLoader((ode_data, tsteps), batchsize = 3)



pinit = ComponentArray(p)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction(loss_neuralode, adtype)
optprob = Optimization.OptimizationProblem(optf, pinit, train_loader)
using IterTools: ncycle

result_neuralode = Optimization.solve(
    optprob, OptimizationOptimisers.Adam(0.01); callback = callback, epochs = 500, maxiters = 1000)

optprob2 = Optimization.OptimizationProblem(optf, result_neuralode.u, train_loader)

result_neuralode2 = Optimization.solve(
    optprob2, OptimizationOptimisers.Lion(0.00005, (0.9, 0.999)); callback, epochs = 300, maxiters = 1000)

# optprob = remake(optprob2; u0 = result_neuralode2.u)



using Serialization
serialize("./models/2_communities_joining/batching-14-08-2025-L.jls", result_neuralode.u)

