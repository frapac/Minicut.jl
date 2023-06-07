@testset "Plots" begin
    iter = 50
    T = 10
    times = rand(iter)
    lowerbounds = rand(iter, T) .- 1
    upperbounds = rand(iter, T) .+ 1.1
    Minicut.gap_plot(lowerbounds, upperbounds)

    Minicut.time_plot(times, lowerbounds[:, 1])
end