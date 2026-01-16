@testset "TemporalNetworkEmbedding" begin

    @testset "Construction and basic properties" begin
        n, d, timesteps = 5, 2, 10
        AL = randn(Float32, n, d, timesteps)
        AR = randn(Float32, n, d, timesteps)

        tne = TemporalNetworkEmbedding(AL, AR, n, d)

        @test tne.n == n
        @test tne.d == d
        @test length(tne) == timesteps
        @test firstindex(tne) == 1
        @test lastindex(tne) == timesteps
    end

    @testset "Dimension mismatch errors" begin
        AL = randn(Float32, 5, 2, 10)
        AR = randn(Float32, 5, 3, 10)  # Different d

        @test_throws DimensionMismatch TemporalNetworkEmbedding(AL, AR, 5, 2)
    end

    @testset "Integer indexing" begin
        n, d, timesteps = 4, 2, 5
        AL = randn(Float32, n, d, timesteps)
        AR = randn(Float32, n, d, timesteps)
        tne = TemporalNetworkEmbedding(AL, AR, n, d)

        # Dict indexing
        frame = tne[3]
        @test frame isa Dict
        @test haskey(frame, :AL)
        @test haskey(frame, :AR)
        @test size(frame[:AL]) == (n, d)

        # Side-specific indexing
        @test tne[3, :AL] == AL[:, :, 3]
        @test tne[3, :AR] == AR[:, :, 3]
    end

    @testset "Float indexing (interpolation)" begin
        n, d = 3, 2
        AL = zeros(Float32, n, d, 3)
        AR = zeros(Float32, n, d, 3)

        # Set up simple values for testing interpolation
        AL[:, :, 1] .= 0.0
        AL[:, :, 2] .= 1.0
        AL[:, :, 3] .= 2.0

        tne = TemporalNetworkEmbedding(AL, AR, n, d)

        # Exact integer should return that frame
        @test tne[2.0, :AL] ≈ AL[:, :, 2]

        # Midpoint should be average
        @test tne[1.5, :AL] ≈ fill(0.5f0, n, d)

        # Quarter point
        @test tne[1.25, :AL] ≈ fill(0.25f0, n, d)
    end

    @testset "Range indexing" begin
        n, d, timesteps = 5, 2, 10
        AL = randn(Float32, n, d, timesteps)
        AR = randn(Float32, n, d, timesteps)
        tne = TemporalNetworkEmbedding(AL, AR, n, d)

        subset = tne[3:7]

        @test subset isa TemporalNetworkEmbedding
        @test length(subset) == 5
        @test subset.n == n
        @test subset.d == d
        @test subset[1, :AL] == tne[3, :AL]
    end

    @testset "without_node" begin
        n, d, timesteps = 5, 2, 10
        AL = randn(Float32, n, d, timesteps)
        AR = randn(Float32, n, d, timesteps)
        tne = TemporalNetworkEmbedding(AL, AR, n, d)

        # Remove node 3
        reduced = without_node(tne, 3)

        @test reduced.n == n - 1
        @test reduced.d == d
        @test length(reduced) == timesteps

        # Check that node 3 is actually removed
        @test reduced[1, :AL][1, :] == tne[1, :AL][1, :]  # Node 1 unchanged
        @test reduced[1, :AL][2, :] == tne[1, :AL][2, :]  # Node 2 unchanged
        @test reduced[1, :AL][3, :] == tne[1, :AL][4, :]  # Now contains old node 4
    end

    @testset "target_node" begin
        n, d, timesteps = 5, 2, 10
        AL = randn(Float32, n, d, timesteps)
        AR = randn(Float32, n, d, timesteps)
        tne = TemporalNetworkEmbedding(AL, AR, n, d)

        # Extract node 3
        target = target_node(tne, 3)

        @test target.n == 1
        @test target.d == d
        @test length(target) == timesteps
        @test target[1, :AL] == reshape(tne[1, :AL][3, :], 1, d)
    end

    @testset "Construction from adjacency matrices" begin
        # Create simple temporal sequence of graphs
        graphs = [
            Float64[0 1 0; 1 0 1; 0 1 0],
            Float64[0 1 1; 1 0 1; 1 1 0],
            Float64[1 1 1; 1 1 1; 1 1 1] .- I(3)
        ]

        tne = TemporalNetworkEmbedding(graphs, 2)

        @test tne.n == 3
        @test tne.d == 2
        @test length(tne) == 3

        # Check embeddings are Float32
        @test eltype(tne.AL) == Float32
        @test eltype(tne.AR) == Float32
    end

end
