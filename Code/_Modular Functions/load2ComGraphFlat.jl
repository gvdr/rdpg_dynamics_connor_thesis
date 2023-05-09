using CSV, DataFrames, LinearAlgebra, Graphs, Tables, DotProductGraphs
include("../_Modular Functions/helperFunctions.jl")

function load2ComGraphFlat(dims)
    path = "./Thesis/Code/Graph Series/$net_name"
    names = readdir(path)

    n, d = dims

    time_graphs = []
    for i in 1:length(names)
        temp = CSV.read(string(path, "/step ", i, ".csv"), DataFrame)
        push!(time_graphs, Matrix(temp[:,:]))
    end

    true_data = zeros(Float32, (d,n), length(names))

    
    tempL, tempR = do_the_rdpg(time_graphs[1], convert(Int, d))
    for i in 1:length(names)
        if i != 1
            L, R = do_the_rdpg(time_graphs[i], convert(Int, d))
            rotated_ortho_procrustes!(L,tempL)
            rotated_ortho_procrustes!(R,tempR)
            tempL, tempR = L, R
        else
            L, R = do_the_rdpg(time_graphs[i], convert(Int, d))
        end
        true_data[:,i] = reshape([L; R], d*n)

    end

    return true_data, time_graphs
end

function loadManualAligned()
    println("Procrustes Alignment")
    path = "./Code/Graph Series/$net_name"
    names = readdir(path)

    n, d = dims

    time_graphs = []
    for i in 1:length(names)
        temp = CSV.read(string(path, "/step", i, ".csv"), DataFrame, header=false)
        push!(time_graphs, Matrix(temp[:,:]))
    end

    true_data = zeros(Float32, d,n, length(names))

    tempL, tempR = do_the_rdpg(time_graphs[1], convert(Int, d))
    for i in 1:length(names)
        if i != 1
            filled(M) = [zeros(Float32, (1,d));M]
            L, R = do_the_rdpg(time_graphs[i], convert(Int, d))
            TL = ortho_procrustes_RM(L',tempL')
            # TR = ortho_procrustes_RM(R',tempR')
            L .= L*TL
            R .= R*TL
            tempL, tempR = L, R
        else
            L, R = do_the_rdpg(time_graphs[i], convert(Int, d))
        end
        true_data[:,:,i] = [L; R]'

    end
    return true_data, time_graphs
end

function loadRegAligned()
    println("Regular Alignment")
    path = "./Code/Graph Series/$net_name"
    names = readdir(path)

    n, d = dims

    time_graphs = []
    for i in 1:length(names)
        temp = CSV.read(string(path, "/step", i, ".csv"), DataFrame, header=false)
        push!(time_graphs, Matrix(temp[:,:]))
    end

    true_data = zeros(Float32, d,n, length(names))

    for i in 1:length(names)
        L, R = do_the_rdpg(time_graphs[i], convert(Int, d))
        true_data[:,:,i] = [L; R]'

    end
    return true_data, time_graphs
end


load2ComGraphFlat(aligned) = aligned ? loadManualAligned() : loadRegAligned()

