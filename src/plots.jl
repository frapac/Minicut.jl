function gap_plot(lowerbounds, upperbounds)
    T = length(lowerbounds[1,:])
    itermax = length(lowerbounds[:,1])
    anim = @animate for i in 1:itermax
        plot(
            1:T,
            [upperbounds[i,:], lowerbounds[i,:]],
            title = "Evaluation along primal trajectory, iter = $i",
            xlabel = "Time step t",
            ylabel = "Value",
            label = ["Upperbounds" "Lowerbounds"]
        )
    end
    gif(anim, "gap_plot.gif", fps = 2)
end

function time_plot(times, values)
    p = plot(cumsum(times), values, xlabel = "Time step t", ylabel = "Value", legend = false)
    savefig(p, "time_plot.gif")
end

function iter_plot(iterations, values)
    p = plot(
        iterations[1,:],
        values[1,:],
        xlabel = "Iteration",
        ylabel = "Value",
    )
end