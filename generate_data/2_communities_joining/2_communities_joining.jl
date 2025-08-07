using LinearAlgebra
using BlockDiagonals
using Arpack
using DotProductGraphs
using Distances
using JSON


include("create_series.jl")
include("../L_R_series.jl")
tsteps = 30

n = 6 # Total number of nodes in the network
d = 2 # Number of dimentions to find for SVD
m = 5

series = create_series(m, n, tsteps)
L_series, R_series = L_R_series(d, tsteps) # Note that this version no longer uses ortho procrustes, as Dot Product reconstruction is much worse, with little improvement in cos distances. 


series[]

i=22
series[i]
L_series[i]*R_series[i]
ltest = [(L_series[i]*[1 0 ; 0 -1 ])' for i in eachindex(L_series)]
rtest = [R_series[i]' for i in eachindex(L_series)]

L_R_translation = []
for i in eachindex(L_series)
    if L_series[i] ≈ R_series[i]
        push!(L_R_translation, [1 0; 0 1])
    elseif L_series[i]*[1 0; 0 -1] ≈ R_series[i]
        push!(L_R_translation, [1 0; 0 -1])
    end
end


s = ltest.≈rtest
for r in s
    println(r)
end


data_dict = Dict("graph_series"=>series, "L_series"=>L_series, "R_series"=>R_series)


open("2_communities_joining.json","w") do f
    JSON.print(f, data_dict)
end
