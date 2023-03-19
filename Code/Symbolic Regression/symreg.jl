using SymbolicRegression
using DelimitedFiles
include("../_Modular Functions/helperFunctions.jl")
include("../_Modular Functions/loadGlobalConstants.jl")

sltn = readdlm("./Code/Solutions/$net_name.csv", ',')
sltnt = readdlm("./Code/Solutions/$net_name test only.csv", ',')
grad_sltn = sltn[:,2:end].-sltn[:,1:end-1]

<<<<<<< HEAD
trace = scatter(x=sltn[1,1:datasize*100],y=sltn[2,1:datasize*100], mode="markers", name="sltn from start")
trace2 =  scatter(x=[t_data[i][1,1] for i in 1:40],y=[t_data[i][2,1] for i in 1:datasize], mode="markers", name="train")
trace3 =  scatter(x=[t_data[i+datasize][1,1] for i in 1:datasize],y=[t_data[i+datasize][2,1] for i in 1:datasize], mode="markers", name="test")
trace4 =  scatter(x=sltnt[1,1:end],y=sltnt[2,1:end], mode="markers", name="sltn from test")

plot([trace, trace2, trace3, trace4])
=======
# trace = scatter(x=sltn[1,1:datasize*100],y=sltn[2,1:datasize*100], mode="markers", name="sltn from start")
# trace2 =  scatter(x=[t_data[i][1,1] for i in 1:40],y=[t_data[i][1,2] for i in 1:datasize], mode="markers", name="train")
# trace3 =  scatter(x=[t_data[i+datasize][1,1] for i in 1:datasize],y=[t_data[i+datasize][1,2] for i in 1:datasize], mode="markers", name="test")
# trace4 =  scatter(x=sltnt[1,1:end],y=sltnt[2,1:end], mode="markers", name="sltn from test")

# plot([trace, trace2, trace3, trace4])
>>>>>>> master

train_data = withoutNode(t_data,1)

function dists(u,t)
    M = train_data[t]

    subtract_func(m) = m-u
    direction_vecs = [subtract_func(m) for m in eachcol(M)]
  
    û = vcat(partialsort(direction_vecs,1:k, by=x->sum(abs2, x))...)
end

iter_sltn = [sltn[:,i] for i in 1:size(sltn)[2]]
data = dists.(iter_sltn, 1.0:0.01:Float64(length(t_data)))
data = hcat(data...)

options = SymbolicRegression.Options(
    binary_operators=[+, *, /, -],
    npopulations=20
)
halls = [
    EquationSearch(
    data[:,1:datasize*100], grad_sltn[i,1:datasize*100], niterations=40, options=options,
    parallelism=:multithreading
    )
    for i in 1:dims[2]
]
# hall_of_fame1 = EquationSearch(
#     data[:,1:datasize*100], grad_sltn[1,1:datasize*100], niterations=40, options=options,
#     parallelism=:multithreading
# )

# hall_of_fame2 = EquationSearch(
#     data[:,1:datasize*100], grad_sltn[2,1:datasize*100], niterations=40, options=options,
#     parallelism=:multithreading
# )

<<<<<<< HEAD
# hall_of_fame3 = EquationSearch(
#     data[:,1:datasize*100], grad_sltn[3,1:datasize*100], niterations=40, options=options,
#     parallelism=:multithreading
# )


dominatings = [
    calculate_pareto_frontier(Float64.(data), grad_sltn[i,:], halls[i], options)
    for i in 1:dims[2]
]

# dominating1 = calculate_pareto_frontier(Float64.(data), grad_sltn[1,:], hall_of_fame1, options)

# dominating2 = calculate_pareto_frontier(Float64.(data), grad_sltn[2,:], hall_of_fame2, options)

# dominating3 = calculate_pareto_frontier(Float64.(data), grad_sltn[3,:], hall_of_fame3, options)
=======
# dominating3 = calculate_pareto_frontier(Float64.(train_data.A), grad_sltn[3,:], hall_of_fame3, options)
>>>>>>> master


# eqn1 = node_to_symbolic(dominating1[end].tree, options)

# eqn2 = node_to_symbolic(dominating2[end].tree, options)

# eqn3 = node_to_symbolic(dominating3[end].tree, options)

<<<<<<< HEAD
# sltn1 = [dominating1[end].tree(data)'; dominating2[end].tree(data)'; dominating3[end].tree(data)']

sltn1 = vcat([dominatings[i][end].tree(data)' for i in 1:dims[2]]...)

=======
sltn1 = [dominating1[end].tree(data)'; dominating2[end].tree(data)']# ; dominating3[end].tree(train_data.A)']

>>>>>>> master
sltn1 = vcat([sum([sltn1[:,j]' for j in 100*(i-1)+1:100*i]) for i in 1:2*datasize-1]...)'



sltn_sym_reg = sltn1[:, datasize+1:2*datasize-1]


temp = zeros(Float64, (dims[2],datasize))
global u0 = targetNode(t_data,1)[1+datasize]
temp[:,1].=u0

for i in 2:datasize
    temp[:,i] = temp[:,i-1]+sltn_sym_reg[:,i-1]
end

sltn_sym_reg = temp



# trace = scatter(x=sltn[1,1:datasize],y=sltn[2,1:datasize], mode="markers")
# trace2 =  scatter(x=[t_data[i][1,1] for i in 1:datasize],y=[t_data[i][1,2] for i in 1:datasize], mode="markers")
# plot([trace, trace2])