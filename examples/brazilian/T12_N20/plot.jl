using Plots, CSV, DataFrames

M_v = Matrix(DataFrame(CSV.File("examples/brazilian/T12_N20/primal/brazilian_sddp_lb.csv"))[1:end, 2:end])
M_d = Matrix(DataFrame(CSV.File("examples/brazilian/T12_N20/dual/brazilian_dualsddp_ub.csv"))[1:end, 2:end])
M_r_l = Matrix(DataFrame(CSV.File("examples/brazilian/T12_N20/reg200/brazilian_regsddp_lb_2.csv"))[1:end, 2:end]) 
M_r_u = Matrix(DataFrame(CSV.File("examples/brazilian/T12_N20/reg200/brazilian_regsddp_ub_2.csv"))[1:end, 2:end]) 
#= 
# V & D after 200 iter. 
# The update rule was
    if relative_gap < 0.1
        mixing = 0.9
        ℓ = mixing * lb + (1.0 - mixing) * ub
    else
        ℓ = lb #mixing * lb + (1.0 - mixing) * ub
    end
=#


start = 1
Max = 500
late_start = 200
n_v = min(length(M_v[:, 1]), Max)
n_d = min(length(M_d[:,1]), Max)
n_r = min(length(M_r_l[:,1]), Max - late_start)

# # Bounds per iteration
# plot(start:n_v, M_v[start:n_v, 1], title="T=12, N=20,t=0", label = "sddp", ylimits=(5e7,8e7), lc=:black)
# plot!(start:n_d, M_d[start:n_d, 1], label = "Dsddp")
# plot!((late_start+1):(late_start+n_r), M_r_l[1:n_r, 1], label = "Reg, a = 10", lc=:green)
# plot!((late_start+1):(late_start+n_r), M_r_u[1:n_r, 1], label = "Reg, a = 10", lc=:green)

# Gap plot along primal traj for regularized sddp
t=11
plot(1:500, log.(M_r_u[:,t].-M_r_l[:, t]), title = "$t")


