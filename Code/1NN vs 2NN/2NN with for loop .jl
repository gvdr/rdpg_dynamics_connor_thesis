using CSV, DataFrames, LinearAlgebra
using PlotlyJS
include("../../_Modular Functions/loadMITGraphFlat.jl")
# User needs to provide:
#   u0
#   datasize
#   tspan
#   ode_data
#   input_data_length (length of one input vector)
#   output_data_length
EPOCHS = 5
const datasize = 10
const dims=(2952,6)   


true_data, time_graphs = loadMITGraphFlat();
true_data = round.(true_data, digits=10);

LR = [reshape(true_data[:,i], (dims[1],dims[2])) for i in 1:size(true_data)[2]]
true_dataL = hcat([reshape(lr[1:Int(dims[1]/2),:], Int(dims[1]/2)*dims[2]) for lr in LR]...)
true_dataR = hcat([reshape(lr[1+Int(dims[1]/2):end,:], Int(dims[1]/2)*dims[2]) for lr in LR]...)


include("../../_Modular Functions/helperFunctions.jl")

ode_dataL = true_dataL[:,1:datasize]
ode_dataR = true_dataR[:,1:datasize]


tspan = (0.0f0, 1.0f0)
tsteps = range(tspan[1], tspan[2], length = 2)
const k = 7

input_data_length = dims[2] + k
output_data_length = dims[2]

# Process Data

# u0::Array{Float64, 1} = Array{Float64, 1}(undef, input_data_length) # Distance of point at ̲0 is always set to 0 to prevent instability
u1::Array{Float64, 1} = Array{Float64, 1}(undef, output_data_length)


include("../../_Modular Functions/constructNN.jl")
include("../../_Modular Functions/NODEproblem.jl")
include("../../_Modular Functions/trainNN.jl")
include("../../_Modular Functions/reconstructRDPG.jl")

prob_neuralodeL, pL, st = constructNN();
prob_neuralodeR, pR, st = constructNN();

optprobL = NODEproblem(pL);
optprobR = NODEproblem(pR);

function predict_neuralode(θ, pnode = prob_neuralodeL)
    Array(solve(pnode, p=θ, save_everystep=false, reltol=1e-5))
end

resultL::AbstractVector = []
for e in 1:EPOCHS
    println("Epoch: ", e)

    for i in 1:datasize
        M::Array{Float64, 2} = reshape(ode_dataL[:,i], (Int(dims[1]/2),dims[2]))
        M̂::Array{Float64, 2} =  reshape(true_dataL[:,i+1], (Int(dims[1]/2),dims[2]))

        @time for v in 1:5#Int(dims[1]/2)
            # iterate over each node/point
            u0 = includeKNNdists(M[v,:], M[1:Int(dims[1]/2).!=v,:])
            u1 = M̂[v, :]
            prob_neuralodeL = remake(prob_neuralodeL, u0=u0)
            resultL = train(optprobL)
            optprobL = remake(optprobL, u0=resultL.u)
            
        end
    end
    println(resultL.u)
    println("")
    println("")
end


function predict_neuralode(θ, pnode = prob_neuralodeR)
    Array(solve(pnode, p=θ, save_everystep=false, reltol=1e-5))
end
resultR::AbstractVector = []
for e in 1:EPOCHS
    println("Epoch: ", e)

    for i in 1:datasize
        M::Array{Float64, 2} = reshape(ode_dataR[:,i], (Int(dims[1]/2),dims[2]))
        M̂::Array{Float64, 2} =  reshape(true_dataR[:,i+1], (Int(dims[1]/2),dims[2]))

        @time for v in 1:5#Int(dims[1]/2)
            # iterate over each node/point
            u0 = includeKNNdists(M[v,:], M[1:Int(dims[1]/2).!=v,:])
            u1 = M̂[v, :]
            prob_neuralodeR = remake(prob_neuralodeR, u0=u0)
            resultR = train(optprobR)
            optprobR = remake(optprobR, u0=resultR.u)
            
        end
    end
    println(resultR.u)
    println("")
    println("")
end



include("../../_Modular Functions/createFullSln.jl")

#create_loss_plot(true_data, prob_neuralode)

