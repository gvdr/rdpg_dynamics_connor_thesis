using Serialization
using JSON
using ComponentArrays

using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
      OptimizationOptimisers, Random, Plots

include("create_forecast.jl")
n = 36
d = 2
# What if u0 is the last observed point at t=10?

function get_preds(data_index, data_path, model_path, n, d)
    data = JSON.parsefile(data_path)
    data = [[L[1] L[2]] for L in data[data_index]] 
    println(data[1])
    forecast, tsteps = create_forecast(model_path, Array(reshape(data[1]', n*d)), n, d)
    preds = collect.(Array.(reshape.(forecast, d,n))')
    return preds, data, tsteps
end

L_preds, L_data, tsteps = get_preds("L_series", "data/long_tail.json", "models/long_tail/batching-14-08-2025-L.jls", n, d)
R_preds, R_data, tsteps = get_preds("R_series", "data/long_tail.json", "models/long_tail/batching-14-08-2025-L.jls", n, d)

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



loss_seq = []
for i in 1:30
    loss_matrix = L_preds[i]*(L_preds[i]*L_R_translation[i])'.-L_data[i]*(L_data[i]*L_R_translation[i])'
    # println(round.(abs.(loss_matrix[1,:]),sigdigits=2))
    push!(loss_seq, sum(abs.(loss_matrix)))
end




gr()
L_trace_d = hcat([d[1,:] for d in L_data]...)
L_trace_p = hcat([p[1,:] for p in L_preds]...)
plt = plot(tsteps,  L_trace_d[2,:]; label = "data", title="Long Tail d2, moving point prediction", xaxis="Time Step")
plot!(plt, tsteps,  L_trace_p[2,:]; label = "prediction")
plot!([15; 15], [-0.8; 8.0], lw=0.5, lc=:red; label = "End of Training Data")
display(plot(plt))


plt = plot(tsteps[1:20],  loss_seq[1:20]; title="Long Tail total RDPG loss", label="loss", xaxis="Time Step", yaxis="Loss")
plot!([15; 15], [0; 1500], lw=0.5, lc=:red; label="end of training data")
