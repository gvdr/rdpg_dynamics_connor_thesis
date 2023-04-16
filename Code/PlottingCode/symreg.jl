using SymbolicRegression
using SymbolicUtils
using DelimitedFiles
include("../_Modular Functions/helperFunctions.jl")

sltn = readdlm("./Code/Solutions/$net_name big net.csv", ',')
sltnt = readdlm("./Code/Solutions/$net_name big net test only.csv", ',')
grad_sltn = sltn[:,2:end].-sltn[:,1:end-1]

test_range = eachindex(1.0:0.01:Float64(datasize))

trace = scatter(x=sltn[1,:],y=sltn[2,:], mode="markers", name="sltn from start")
trace2 =  scatter(x=[t_data[i][1,1] for i in 1:datasize],y=[t_data[i][2,1] for i in 1:datasize], mode="markers", name="train")
trace3 =  scatter(x=[t_data[i+datasize][1,1] for i in 1:length(t_data)-datasize],y=[t_data[i+datasize][2,1] for i in 1:length(t_data)-datasize], mode="markers", name="test")
trace4 =  scatter(x=sltnt[1,1:end],y=sltnt[2,1:end], mode="markers", name="sltn from test")

plot([trace, trace2, trace3, trace4])

train_data = withoutNode(t_data,1)
train_data[1]

function dists(u,t)
    M = train_data[t]
    
    subtract_func(m) = m-u
    direction_vecs = [subtract_func(m) for m in eachcol(M)]

    û = vcat(partialsort(direction_vecs,1:k, by=x->sum(abs2, x))...)

end

iter_sltn = [sltn[:,i] for i in 1:size(sltn)[2]]
data = dists.(iter_sltn, 1.0:0.01:Float64(datasize))
data = hcat(data...)

options = SymbolicRegression.Options(
    binary_operators=[+, *, /, -],
    npopulations=20
)
halls = [
    EquationSearch(
    data[:,test_range[1:end-1]], grad_sltn[i,test_range[1:end-1]], niterations=40, options=options,
    parallelism=:multithreading
    )
    for i in 1:dims[2]
]



dominatings = [
    calculate_pareto_frontier(Float64.(data), grad_sltn[i,:], halls[i], options)
    for i in 1:dims[2]
]

eqns = [
    dominatings[i][end-2].tree
    for i in 1:dims[2]
]

sltn1 = vcat([dominatings[i][end].tree for i in 1:dims[2]]...)



function next_step(u0, t, eq)
    d = dists(u0, t+datasize)[:,:]
    u0+eq(d)
end

sltn1 = zeros(Float64, (dims[2],length(1.0:0.01:size(sltnt)[2])))
global u0 = targetNode(t_data,1)[1+datasize]
sltn1[:,1] = u0
iter_eq(d) = vcat([eqns[i](d) for i in 1:dims[2]]...)
for t in 2:size(sltn1)[2]
    sltn1[:,t] = next_step(sltn1[:,t-1], 1+(t-1)*0.01, iter_eq)
end


sltn_sym_reg = sltn1[:, 1:100:size(sltn1)[2]]


# temp = zeros(Float64, (dims[2],datasize))
# global u0 = targetNode(t_data,1)[1+datasize]
# temp[:,1].=u0

# for i in 2:datasize
#     temp[:,i] = temp[:,i-1]+sltn_sym_reg[:,i-1]
# end

# sltn_sym_reg = temp



# trace = scatter(x=sltn[1,1:datasize],y=sltn[2,1:datasize], mode="markers")
# trace2 =  scatter(x=[t_data[i][1,1] for i in 1:datasize],y=[t_data[i][1,2] for i in 1:datasize], mode="markers")
# plot([trace, trace2])