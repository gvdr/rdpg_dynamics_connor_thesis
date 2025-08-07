using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL, OptimizationOptimisers, Random, Plots
using JSON

data = JSON.parsefile("./data/1_community_oscillation.json")

n = 5 # Total Number of Nodes in network
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
datasize = 10
tspan = (0.0f0, 9.0f0)
tsteps = range(tspan[1], tspan[2]; length = datasize)

ode_data = Array(u[:,1:datasize])
data_translation = L_R_translation[1:datasize]

dudt2 = Chain(x -> x, Dense(n*d, 256, celu), Dense(256, 128, celu), Dense(128, 128, celu), Dense(128, n*d))
p, st = Lux.setup(rng, dudt2)

prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(); saveat = tsteps)

function predict_neuralode(p)
    Array(prob_neuralode(u0, p, st)[1])
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

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    reshaped_preds = [reshape(col, n, d) for col in eachcol(pred)]
    pred_arrays = pred_to_array.(reshaped_preds, data_translation)
    less_than_zero_loss = sum(lower_than_zero_loss.(pred_arrays))
    more_than_one_loss = sum(greater_than_one_loss.(pred_arrays))
    return loss + less_than_zero_loss + more_than_one_loss
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
    optprob, OptimizationOptimisers.Adam(0.01); callback = callback, maxiters = 300)

optprob2 = remake(optprob; u0 = result_neuralode.u)

# result_neuralode2 = Optimization.solve(
#     optprob2, Optim.BFGS(; initial_stepnorm = 0.025); callback, allow_f_increases = false, maxiters = 100)

# optprob = remake(optprob2; u0 = result_neuralode2.u)


callback((; u = result_neuralode.u), loss_neuralode(result_neuralode.u); doplot = true)


using Serialization
serialize("./models/1_community_oscillation/big-NN-07-08-2025-L.jls", result_neuralode.u)

