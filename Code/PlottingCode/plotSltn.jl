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
mid = convert(Int, dims[1]/2)

for i in 1:length(t_data)-datasize
    pts = t_data[i+datasize]'#get_embedding([sltn_sym_reg[:,i]'; t_data[i+datasize]], sltn[:,i])
    mid = convert(Int, dims[1]/2)
    traces0 = PlotlyJS.scatter(x=[sltn[1,i]], y=[sltn[2,i]], mode="markers", name="Neural Network Pred", marker_size=12)
    traces1 = PlotlyJS.scatter(x=[sltn_sym_reg[1,i]], y=[sltn_sym_reg[2,i]], mode="markers", name="Symbolic Regression Pred", marker_size=12)
    traces2 = PlotlyJS.scatter(x=[pts[1,1]], y=[pts[1,2]], mode="markers", name="Target Node", marker_size=12)
    
    traces3 = PlotlyJS.scatter(x=pts[2:mid,1], y=pts[1:mid,2], mode="markers", name="Data")

    layout = Layout(xaxis=attr(showgrid=false),yaxis=attr(showgrid=false))

    display(PlotlyJS.plot([traces0,traces1,traces2, traces3],layout))
    savefig(PlotlyJS.plot([traces0, traces1, traces2, traces3],layout), "./Code/Plots/Test Only/$net_name/$net_name $i small net.png")

end




function net_pred_loss(p, pt, tsteps)
    # p ∈ s, n, t
    # pR' = a
    # round(a)
    true_struct = time_graphs[tstep][1,2:end]

    Rs = [t_data[t][:,mid+2:end] for t in tsteps]

    
    pred = (p-pt)'.* Rs
    pred = vcat(pred...)
    println(typeof(pred))
    return sum(abs2, pred)/length(tsteps),pred,true_struct
end

tsteprange = datasize+5:datasize+8
sltn_sym = sltn_sym_reg[:,tstep-datasize]
sltn_nn = sltn[:,tstep-datasize]
sltn_true = t_data[datasize+1][:,1]

pt=sltn_true
p = [sltn_nn, sltn_sym]
l  = net_pred_loss.(p,[pt,pt],[tsteprange])

L = t_data[tstep][:,1:mid]
R = t_data[tstep][:,mid+1:end]

L'*R
(p[1][1], p[2][1], p[3][1])
printall(p[1][3])