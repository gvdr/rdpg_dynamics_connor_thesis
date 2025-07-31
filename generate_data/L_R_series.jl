function L_R_series(d, tsteps)
    L_series= []
    R_series = []

    function svd_decompose(A,d)
        L,Σ,R = svds(A; nsv=d, v0=[Float64(i%7) for i in 1:minimum(size(A))])[1]
        println(Σ)
        L̂ = L * diagm(.√Σ)
        R̂ = R * diagm(.√Σ)
        return (L̂ = L̂, R̂ = R̂)
    end

    for i in 1:tsteps
        L, R = svd_decompose(series[i], convert(Int, d))
        push!(L_series, L)
        push!(R_series, R)

    end
    return L_series, R_series
end