@testset "Constraint Functions" begin

    @testset "below_zero_penalty" begin
        # Matrix with no negative values
        P = [0.5 0.3; 0.7 0.5]
        @test below_zero_penalty(P) == 0.0

        # Matrix with negative off-diagonal
        P = [0.5 -0.2; 0.7 0.5]
        @test below_zero_penalty(P) ≈ 0.2

        # Multiple negative values
        P = [0.5 -0.1 -0.3; 0.2 0.5 -0.2; 0.1 0.4 0.5]
        @test below_zero_penalty(P) ≈ 0.1 + 0.3 + 0.2

        # Negative diagonal should be ignored (with exclude_diagonal=true)
        P = [-0.5 0.3; 0.7 -0.5]
        @test below_zero_penalty(P; exclude_diagonal=true) == 0.0

        # Include diagonal
        @test below_zero_penalty(P; exclude_diagonal=false) ≈ 1.0
    end

    @testset "above_one_penalty" begin
        # Matrix with no values above 1
        P = [0.5 0.3; 0.7 0.5]
        @test above_one_penalty(P) == 0.0

        # Matrix with value above 1
        P = [0.5 1.2; 0.7 0.5]
        @test above_one_penalty(P) ≈ 0.2

        # Multiple values above 1
        P = [0.5 1.1 1.5; 0.2 0.5 1.3; 0.1 0.4 0.5]
        @test above_one_penalty(P) ≈ 0.1 + 0.5 + 0.3

        # Diagonal above 1 should be ignored
        P = [1.5 0.3; 0.7 1.5]
        @test above_one_penalty(P; exclude_diagonal=true) == 0.0
    end

    @testset "probability_constraint_loss" begin
        # Valid probability matrix
        P = [0.5 0.3 0.2; 0.4 0.5 0.6; 0.1 0.7 0.5]
        @test probability_constraint_loss(P) == 0.0

        # Mixed violations
        P = [0.5 -0.1 1.2; 0.3 0.5 0.8; 0.9 0.1 0.5]
        expected = 0.1 + 0.2  # -0.1 below zero, 1.2-1=0.2 above one
        @test probability_constraint_loss(P) ≈ expected

        # Custom weights
        @test probability_constraint_loss(P; lambda_lower=2.0, lambda_upper=0.5) ≈ 2.0 * 0.1 + 0.5 * 0.2
    end

    @testset "probability_constraint_loss on vector of matrices" begin
        P1 = [0.5 -0.1; 0.3 0.5]  # One violation: -0.1
        P2 = [0.5 1.2; 0.8 0.5]   # One violation: 1.2

        total = probability_constraint_loss([P1, P2])
        @test total ≈ 0.1 + 0.2
    end

    @testset "Type stability" begin
        P32 = Float32[0.5 -0.1; 1.2 0.5]
        P64 = Float64[0.5 -0.1; 1.2 0.5]

        @test below_zero_penalty(P32) isa Float32
        @test below_zero_penalty(P64) isa Float64

        @test probability_constraint_loss(P32) isa Float32
        @test probability_constraint_loss(P64) isa Float64
    end

end
