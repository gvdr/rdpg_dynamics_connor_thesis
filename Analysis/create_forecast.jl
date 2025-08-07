using Serialization


function create_forecast(MODEL_PATH, u0, n, d)
    
    u = deserialize(MODEL_PATH)
    datasize = 30
    tspan = (0.0f0, 29.0f0)
    tsteps = range(tspan[1], tspan[2]; length = datasize)


    rng = Xoshiro(0)

    dudt2 = Chain(x -> x, Dense(n*d, 256, celu), Dense(256, 128, celu), Dense(128, 128, celu), Dense(128, n*d))
    p, st = Lux.setup(rng, dudt2)
    prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(); saveat = tsteps)

    function predict_neuralode(p)
        Array(prob_neuralode(u0, p, st)[1])
    end

    array_pred = predict_neuralode(u)

    return [array_pred[:,i] for i in 1:size(array_pred, 2)], tsteps
end
