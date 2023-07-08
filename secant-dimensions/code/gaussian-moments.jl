using LinearAlgebra

# ------------------------------------------------
#   Sampling
# ------------------------------------------------

random_quadratic = function(variables)
    qmon = monomials(variables, 2);
    return randn(length(qmon))⋅qmon; 
end

random_lfsquares = function(variables, amount::Int64)
    return [(randn(length(variables))⋅variables)^2 for i=1:amount];
end

random_quadratics = function(variables, amount::Int64)
    return [random_quadratic(X) for i=1:amount];
end


# ------------------------------------------------
#  Tangent Space of Gaussian Moment Variety
# ------------------------------------------------

#df_gaussians = DataFrame(CSV.read("data/gaussian_moment_derivatives.csv"), DataFrame) 
# gaussian_moments(q, ℓ, degree) = df_gaussians.moments[degree]; # seems like a way to get into performance trouble...

# we assume that f and g and d are global constants defined in the jupyter notebook
f_ideal = function(q, ℓ, vars)
    monoms_d = monomials(vars, d)
    monoms_2 = monomials(vars, 2)
    coeffs = [sparse(coefficients(f(q, ℓ)*q_mon, monoms_d)) for q_mon in monoms_2]
    return hcat(coeffs...)
end

g_ideal = function(q, ℓ, vars)
    monoms_d = monomials(vars, d)
    coeffs = [sparse(coefficients(g(q, ℓ)*var, monoms_d)) for var in vars]
    return hcat(coeffs...)
end

secant_tangent_dimension_general = function(n::Int, vars, m=nothing)
    M = length(monomials(vars, 1:2));
    N = length(monomials(vars, d));
    m = (m === nothing) ? Int(floor(N/M)) : m;
    if m == 0
        return 0, 0, m, sparse(Matrix(I, 0, 0)), sparse(Matrix(I, 0, 0))
    end
    q̄ = random_quadratics(vars, m)
    ℓ̄ = [vars⋅randn(n) for i=1:m];
    A = hcat([f_ideal(q, ℓ, vars) for (q, ℓ) ∈ zip(q̄, ℓ̄)]...)
    B = hcat([g_ideal(q, ℓ, vars) for (q, ℓ) ∈ zip(q̄, ℓ̄)]...)
    dim = rank(hcat(A, B))
    exp_dim = m*M
    return dim, exp_dim, m, A, B
end

secant_tangent_dimension_general_lfsquared = function(n::Int, vars, m=nothing)
    M = length(monomials(vars, 1:2));
    N = length(monomials(vars, d));
    m = (m === nothing) ? Int(floor(N/M)) : m;
    if m == 0
        return 0, 0, m, sparse(Matrix(I, 0, 0)), sparse(Matrix(I, 0, 0))
    end
    q̄ = random_lfsquares(vars, m)
    ℓ̄ = [vars⋅randn(n) for i=1:m];
    A = hcat([f_ideal(q, ℓ, vars) for (q, ℓ) ∈ zip(q̄, ℓ̄)]...)
    B = hcat([g_ideal(q, ℓ, vars) for (q, ℓ) ∈ zip(q̄, ℓ̄)]...)
    dim = rank(hcat(A, B))
    exp_dim = m*M
    return dim, exp_dim, m, A, B
end

secant_tangent_dimension_specific = function(n, vars, q̄, ℓ̄)
    m = length(q̄);
    M = length(monomials(vars, 1:2));
    A = hcat([f_ideal(q, ℓ, vars) for (q, ℓ) ∈ zip(q̄, ℓ̄)]...)
    B = hcat([g_ideal(q, ℓ, vars) for (q, ℓ) ∈ zip(q̄, ℓ̄)]...)
    dim = rank(Matrix(hcat(A, B)))
    exp_dim = m*M
    return dim, exp_dim, m, A, B
end


