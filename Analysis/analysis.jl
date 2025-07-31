using Serialization
using JSON
using ComponentArrays

using ComponentArrays, Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
      OptimizationOptimisers, Random, Plots

include("create_forecast.jl")


function get_preds(data_index, data_path, model_path)
    data = JSON.parsefile(data_path)
    data = [[L[1] L[2]] for L in data[data_index]] 

    forecast, tsteps = create_forecast(model_path, Array(reshape(data[1]', 10)))
    preds = collect.(Array.(reshape.(forecast, 2,5))')
    return preds, data, tsteps
end

L_preds, L_data, tsteps = get_preds("L_series", "data/1_community_oscillation.json", "models/1_community_oscillation/25-07-2025-L.jls")
R_preds, R_data, tsteps = get_preds("R_series", "data/1_community_oscillation.json", "models/1_community_oscillation/25-07-2025-R.jls")



for i in 1:30
    loss_matrix = L_preds[i]*(L_preds[i]*[1 0; 0 -1])'.-L_data[i]*(L_data[i]*[1 0; 0 -1])'
    # println(round.(abs.(loss_matrix[1,:]),sigdigits=2))
    println(loss_matrix)
    println("")
end

gr()

L_trace_d = hcat([d[1,:] for d in L_data]...)
L_trace_p = hcat([p[1,:] for p in L_preds]...)
plt = plot(1:tsteps, L_trace_d[1,:], L_trace_d[2,:]; label = "data")
plot!(plt, 1:tsteps, L_trace_p[1,:], L_trace_p[2,:]; label = "prediction")
display(plot(plt))
