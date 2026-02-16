@testset "Embedding Functions" begin

    @testset "Orthogonal Procrustes" begin
        # Test that result is orthogonal
        A = randn(3, 5)
        B = randn(3, 5)
        Omega = ortho_procrustes_RM(A, B)

        @test size(Omega) == (3, 3)
        @test Omega * Omega' ≈ I(3) atol=1e-10
        @test Omega' * Omega ≈ I(3) atol=1e-10

        # Test that aligned A is closer to B
        aligned_A = Omega * A
        dist_before = norm(A - B)
        dist_after = norm(aligned_A - B)
        @test dist_after <= dist_before + 1e-10
    end

    @testset "Truncated SVD" begin
        # Create a low-rank matrix
        A = randn(10, 10)
        A = A + A'  # Make symmetric

        U, S, V = truncated_svd(A, 3)

        @test size(U) == (10, 3)
        @test size(V) == (10, 3)
        @test length(S) == 3

        # Singular values should be positive and sorted descending
        @test all(S .>= 0)
        @test issorted(S, rev=true)

        # Reconstruction should be reasonable
        A_approx = U * Diagonal(S) * V'
        @test norm(A - A_approx) < norm(A)  # Should capture some variance
    end

    @testset "SVD Embedding" begin
        # Simple symmetric adjacency matrix (triangle graph)
        A = Float64[0 1 1; 1 0 1; 1 1 0]

        emb = svd_embedding(A, 2)

        @test size(emb.L_hat) == (3, 2)
        @test size(emb.R_hat) == (3, 2)

        # Reconstruction: L * R' should approximate A
        A_reconstructed = emb.L_hat * emb.R_hat'
        @test norm(A_reconstructed - A) < 1.0  # Reasonable approximation
    end

    @testset "SVD Embedding with custom engine" begin
        A = Float64[0 1 0 1; 1 0 1 0; 0 1 0 1; 1 0 1 0]

        # Custom engine using full SVD
        custom_svd(M, d) = begin
            F = svd(M)
            (F.U[:, 1:d], F.S[1:d], F.V[:, 1:d])
        end

        emb = svd_embedding(A, custom_svd, 2)

        @test size(emb.L_hat) == (4, 2)
        @test size(emb.R_hat) == (4, 2)
    end

end
