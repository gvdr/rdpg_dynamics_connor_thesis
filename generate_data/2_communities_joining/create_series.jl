function create_series(m::Int, n::Int, tsteps::Int)
    A = Matrix(BlockDiagonal([ones(m,m), ones(n,n)]))
    for i in 1:n+m
        A[i, i] = 0
    end

    series = [copy(A)]

    for i in 1:m
        for j in 1:n
            A = copy(series[end])
            A[i+m, j] = 1

            A[j, i+m] = 1


        push!(series, copy(A)) 
        end
    end

    return series[1:tsteps]
end

