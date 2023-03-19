using Revise
<<<<<<< HEAD
global dims=(212,2) #|> Lux.gpu
global net_name = "3community"
=======
global dims=(182,2) #|> Lux.gpu
global net_name = "longTail"
>>>>>>> master
include("load2ComGraphFlat.jl")
include("../structs/TemporalEmbedding.jl")
global true_data, time_graphs = load2ComGraphFlat(true);
Base.IndexStyle(true_data)=IndexLinear()

t_data = TemporalNetworkEmbedding(true_data,dims[1],dims[2])


global datasize = 20
global train_data = withoutNode(t_data[1:datasize],1) #|> Lux.gpu
global test_data = withoutNode(t_data[1+datasize:end],1)
global tspan = (1.0, 10.0)#|> Lux.gpu
global tsteps = range(tspan[1], tspan[2], length = datasize)#|> Lux.gpu
global k = 15

global input_data_length = k*dims[2]
global output_data_length = dims[2]
global u0 = vec(targetNode(t_data,1)[1])#|> Lux.gpu


#targetNode(t_data,1)[1]
# TNode_data = targetNode(t_data,1)
# train_data = withoutNode(t_data,1)
# function foo(u,t)
#     M = reshape(train_data.A[:,Int(floor(t))],(train_data.n,train_data.d))
        
#     cosine_distances = zeros(Float64, train_data.n)

#     for i::Int in eachindex(cosine_distances)
#         distance = dot(M[i,:],u)/ (norm(u) * norm(M[i,:])) #cosine_dist(target, node from data)
#         if typeof(distance) == Float32
#         @inbounds cosine_distances[i] = isnan(distance) ? zero(distance) : distance
#         else
#         @inbounds cosine_distances[i] = isnan(distance.value) ? zero(distance.value) : distance.value
#         end
#     end
#     return partialsort(cosine_distances,1:k, by=x->x)
# end

# us = [TNode_data[i] for i in 1:40]
# in_dists = hcat(foo.(us, 1:40)...)

# in_dists[:,6:10]
# [t_data[i-1][1,2].-t_data[i][1,2] for i in 2:length(t_data)]
# trace=PlotlyJS.scatter(x=t_data[2][:,1],
#         y=t_data[20][:,2],
#         color=1:91,
#         mode="markers", name="Neural Network Pred", marker_size=12)

# plot(trace)