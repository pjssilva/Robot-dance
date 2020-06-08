using JuMP
using Ipopt

function my_recur_model(n)
    m = Model(optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 5, "linear_solver" => "mumps"))
    @variable(m, 0 <= x[1:n] <= 1)
    pre_c = Vector{NonlinearExpression}(undef, n)
    pre_c[1] = @NLexpression(m, (x[1] - 0.5)^2)
    for i = 2:n
        pre_c[i] = @NLexpression(m, pre_c[i - 1] + (x[i] - 0.5)^2)
    end
    @NLconstraint(m, [i =1:n], pre_c[i] == 0)
    @NLobjective(m, Max, sum(x[i]^2 for i = 1:n))
    return m
end


function my_recur_model2(n)
    m = Model(optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 5, "linear_solver" => "mumps"))
    @variable(m, 0 <= x[1:n] <= 1)
    @NLconstraint(m, [i=1:n], sum((x[j] - 0.5)^2 for j = 1:i) == 0)
    @NLobjective(m, Max, sum(x[i]^2 for i = 1:n))
    return m
end


function my_recur_model3(n)
    m = Model(optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 5, "linear_solver" => "mumps"))
    @variable(m, 0 <= x[1:n] <= 1)
    @variable(m, pre_c[1:n])
    @NLconstraint(m, pre_c[1] == (x[1] - 0.5)^2)
    for i = 2:n
        @NLconstraint(m, pre_c[i] == pre_c[i - 1] + (x[i] - 0.5)^2)
    end
    @NLobjective(m, Max, sum(x[i]^2 for i = 1:n))
    @NLconstraint(m, [i =1:n], pre_c[i] == 0)
    return m
end
