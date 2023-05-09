using PlotlyJS
using ColorSchemes
include("../_Modular Functions/loadGlobalConstants.jl")

include("../_Modular Functions/pca.jl")
global_consts("longTail", (182,2))

include("symreg.jl")






function get_embedding(B, pred)
    mid = convert(Int, dims[1]/2)
    L = B[1:mid,:]

    if dims[2] > 2
        return principle_components([pred'; L])
    else
        return [pred'; L]
    end
end

using DelimitedFiles


sltn = readdlm("./Code/Solutions/$net_name big net test only.csv", ',')

for i in 1:length(t_data)-datasize
    pts = t_data[i+datasize]'#get_embedding([sltn_sym_reg[:,i]'; t_data[i+datasize]], sltn[:,i])
    mid = convert(Int, dims[1]/2)
    traces0 = PlotlyJS.scatter(x=[sltn[1,i]], y=[sltn[2,i]], mode="markers", name="Neural Network Pred", marker_size=12)
    traces1 = PlotlyJS.scatter(x=[sltn_sym_reg[1,i]], y=[sltn_sym_reg[2,i]], mode="markers", name="Symbolic Regression Pred", marker_size=12)
    traces2 = PlotlyJS.scatter(x=[pts[1,1]], y=[pts[1,2]], mode="markers", name="Target Node", marker_size=12)
    
    traces3 = PlotlyJS.scatter(x=pts[2:mid,1], y=pts[1:mid,2], mode="markers", name="Data")
    display(PlotlyJS.plot([traces0,traces1,traces2, traces3]))
    savefig(PlotlyJS.plot([traces0, traces1, traces2, traces3]), "./Code/Plots/Test Only/$net_name/$net_name $i small net.png")

end




function net_pred_loss(pt, tstep)
    # p ∈ s, n, t
    # pR' = a
    # round(a)
    true_struct = time_graphs[tstep][1,2:end]

    R = t_data[tstep][:,mid+2:end]

    
    pred = round.(pt'*R)
    # 
    return sum(abs, pred.-true_struct'),pred
end

tstep = datasize+15
sltn_sym = sltn_sym_reg[:,tstep-datasize]
sltn_nn = sltn[:,tstep-datasize]
sltn_true = t_data[datasize+1][:,1]

tests = [sltn_sym, sltn_nn, sltn_true]
p  = net_pred_loss.(tests,tstep)

L = t_data[tstep][:,1:mid]
R = t_data[tstep][:,mid+1:end]

L'*R

printall(p[1][2])