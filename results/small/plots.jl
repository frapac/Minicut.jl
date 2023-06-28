using DataFrames, CSV

iter = 450

x = Matrix(DataFrame(CSV.File("brazilian_runlb_regsddp.csv"))[1:iter, 2:end])
y = Matrix(DataFrame(CSV.File("brazilian_runub_regsddp.csv"))[1:iter, 2:end])

Minicut.gap_plot(x,y)