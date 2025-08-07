using LinearAlgebra
using BlockDiagonals
using Arpack
using DotProductGraphs
using Distances
using JSON


include("create_series.jl")
include("../L_R_series.jl")
tsteps = 30

n = 30 # Length of the tail
d = 2 # Number of dimentions to find for SVD
c = 5 # Size of the community

series = create_series(n, c, tsteps)
L_series, R_series = L_R_series(d, tsteps) # Note that this version no longer uses ortho procrustes, as Dot Product reconstruction is much worse, with little improvement in cos distances. 

i=1
series[i]
L_series[i]*R_series[i]'
ltest = [L_series[i]*(L_series[i]*[1 0 ; 0 -1 ])' for i in eachindex(L_series)]
rtest = [L_series[i]*R_series[i]' for i in eachindex(L_series)]




data_dict = Dict("graph_series"=>series, "L_series"=>L_series, "R_series"=>R_series)


open("long_tail.json","w") do f
    JSON.print(f, data_dict)
end