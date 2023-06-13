using PlotlyJS
using ColorSchemes
using Lux
# using SymbolicRegression
# using SymbolicUtils
using DelimitedFiles
include("../_Modular Functions/loadGlobalConstants.jl")

# include("../_Modular Functions/pca.jl")
global_consts("2communities", (182,2))

include("symreg.jl")
p1, p2 = (1,2)


using DelimitedFiles


sltn = readdlm("./Code/Solutions/$net_name big net test only.csv", ',')

t_data[1][:AL]

for i in 1:length(t_data)
    pts = t_data[i][:AL]#get_embedding([sltn_sym_reg[:,i]'; t_data[i+datasize]], sltn[:,i])
    
    #traces0 = PlotlyJS.scatter(x=[sltn[p1,i]], y=[sltn[p2,i]], mode="markers", name="Neural Network Pred", marker_size=12)
    #traces1 = PlotlyJS.scatter(x=[sltn_sym_reg[p1,i]], y=[sltn_sym_reg[p2,i]], mode="markers", name="Symbolic Regression Pred", marker_size=12)

    traces0 = PlotlyJS.scatter(x=[], y=[], mode="markers", name="Neural Network Pred", marker_size=12)
    traces1 = PlotlyJS.scatter(x=[], y=[], mode="markers", name="Symbolic Regression Pred", marker_size=12)

    traces2 = PlotlyJS.scatter(x=[pts[1,p1]], y=[pts[1,p2]], mode="markers", name="Target Node", marker_size=12)
    
    traces3 = PlotlyJS.scatter(x=pts[2:end,p1], y=pts[2:end,p2], mode="markers", name="Data")

    layout = Layout(xaxis=attr(showgrid=false),yaxis=attr(showgrid=false))

    display(PlotlyJS.plot([traces0,traces1,traces2, traces3],layout))# 
    PlotlyJS.savefig(PlotlyJS.plot([traces0, traces1, traces2, traces3],layout), "./Code/Plots/Test Only/$net_name/$net_name $i illustration.png")

end




function net_pred_loss(p, pt, tsteps)
    # p ∈ s, n, t
    # pR' = a
    # round(a)
    

    Rs = [t_data[t,:AR]' for t in tsteps]

    println(size.(Rs))
    pred = [(p[:,i]-pt[:,i])' * Rs[i] for i in 1:length(tsteps)]
    println(size.(pred))
    pred = vcat(pred...)
    return sum(abs2, pred)/length(tsteps)
end

tsteprange = datasize+5:datasize+8
sltn_sym = sltn_sym_reg[:,tsteprange.-datasize]
sltn_nn = sltn[:,tsteprange.-datasize]
sltn_true = targetNode(t_data, 1)[tsteprange].AL[1,:,:]

pt=sltn_true
p = [sltn_nn, sltn_sym]
l  = net_pred_loss.(p,[pt,pt],[tsteprange])
println(l)