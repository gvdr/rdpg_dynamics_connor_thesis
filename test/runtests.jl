using RDPGDynamics
using Test
using LinearAlgebra

@testset "RDPGDynamics.jl" begin
    include("test_embedding.jl")
    include("test_types.jl")
    include("test_constraints.jl")
end
