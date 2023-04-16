using DiffEqFlux, DifferentialEquations, OptimizationOptimJL, CSV
include("/home/connor/Thesis Project/Toy Example Graph/_Modular Functions/helperFunctions.jl")
function knownODEpoints()
    u0=[0.5 0.5;
        0.125 -0.125;
        -0.5 0.0;
        -0.5 -0.5]
    u0=reshape(u0, 8)
    datasize = 100
    tspan = (0.0f0, 10.0f0)
    tsteps = range(tspan[1], tspan[2], length = datasize)


    function trueODEfunc(du, u, p, t)
    u_ = reshape(u, (4,2))
    du_ = reshape(du, (4,2))
    du_ = (sum.(abs2, eachrow(u_)).-0.25).*[u_[:,2] -u_[:,1]]
    du .= reshape(du_, 8)
    end

    prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
    ode_data= Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

    ode_data_ = reshape(ode_data, (4,2,datasize))
end
