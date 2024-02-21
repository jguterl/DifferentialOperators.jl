using DifferentialOperators
using BenchmarkTools

import LinearAlgebra.norm as norm


function compute_error(f1::ScalarField, f0::ScalarField, g::Grid, intx, inty)
    df = ScalarField(g)
    @. df.field.data = f1.field.data - f0.field.data
    return norm(df.field.data[intx, inty], Inf)
end

function compute_error(v1::VectorField, v0::VectorField, g::Grid, intx, inty)
    dv = VectorField(g)
    @. dv.x.data = v1.x.data - v0.x.data
    @. dv.y.data = v1.y.data - v0.y.data
    @. dv.z.data = v1.z.data - v0.z.data
    return norm(dv.x.data[intx, inty], Inf), norm(dv.y.data[intx, inty], Inf), norm(dv.z.data[intx, inty], Inf)
end

# A wrapper for testing (f is the operator we are interested in)
#test!(f, result, intx, inty)          = compute!(f, grid_data, result, intx, inty)

#test_threads!(f, result, intx, inty)  = compute_threads!(f, grid_data, result, intx, inty)

# Analytical solutions to the problems
kx = 2;
ky = 1;
kz = 3;

@variables x y
∂x = Differential(x)
∂y = Differential(y)

ψ₀ = cos(kx * x) * cos(ky * y)
∂x_ψ₀ = expand_derivatives(∂x(ψ₀))
∂y_ψ₀ = expand_derivatives(∂y(ψ₀))
∂xx_ψ₀ = expand_derivatives(∂x(∂x(ψ₀)))
∂yy_ψ₀ = expand_derivatives(∂y(∂y(ψ₀)))

fx = eval(build_function(∂x_ψ₀, x, y))
fy = eval(build_function(∂y_ψ₀, x, y))
fxx = eval(build_function(∂xx_ψ₀, x, y))
fyy = eval(build_function(∂yy_ψ₀, x, y))

# Grid sizes
Lx, Ly = 2π, 4π
nx0, ny0 = 32, 64
ng = 1
ng2 = 2
nreps = 6

nscltests = 4
nvectests = 4
ndims = 2
nerrs = ndims * nvectests + nscltests
num_err = zeros(nreps, nerrs)

for n = 1:nreps

    nx = nx0 * 2^(n - 1) # Number of interior points
    ny = ny0 * 2^(n - 1) # Number of interior points

    intx = 1+ng:nx+ng
    inty = 1+ng:ny+ng

    intx2 = 1+ng2:nx+ng2
    inty2 = 1+ng2:ny+ng2

    # First we create a grid that start at x=0 on the first interior point
    local_grid = Grid(nx, ny; L=[Lx, Ly], d0=[0.0, 0.0], ng=[1, 1])

    # We also define the data needed to calculated the derivatives (we can define order of accuracy here)
    grid_data = GridDerivatives(local_grid)

    dx = grid_data.dx.data[2, 2]
    dy = grid_data.dy.data[2, 2]

    # Staggered grids asuming half a grid point displacement
    # This is the standard finite volume grid for fluxes through faces
    gridvˣ = Grid(nx, ny; L=[Lx, Ly], d0=[0.5 * dx, 0], ng=[1, 1])
    gridvʸ = Grid(nx, ny; L=[Lx, Lx], d0=[0.0, 0.5 * dy], ng=[1, 1])
    gridvᶻ = Grid(nx, ny; L=[Lx, Ly], d0=[0.0, 0.0], ng=[1, 1]) #Placehold for z grid

    # This is the adjoint grid produced using curl(v) 
    gridbˣ = Grid(nx, ny; L=[Lx, Ly], d0=[0.0, 0.5 * dy], ng=[1, 1]) # 0.5*dz
    gridbʸ = Grid(nx, ny; L=[Lx, Ly], d0=[0.5 * dx, 0.0], ng=[1, 1]) # 0.5*dz
    gridbᶻ = Grid(nx, ny; L=[Lx, Ly], d0=[0.5 * dx, 0.5 * dy], ng=[1, 1]) #Placehold for z grid

    ψ = ScalarField(local_grid)
    @. ψ.field.data = cos(kx * local_grid.x) * cos(ky * local_grid.y)

    V = VectorField(local_grid)
    @. V.x.data = 0.0
    @. V.y.data = 0.0
    @. V.z.data = ψ.field.data

    v_num = VectorField(local_grid)
    s_num = ScalarField(local_grid)
    v_ana = VectorField(local_grid)
    s_ana = ScalarField(local_grid)

    #
    # ∇ψ, ∇²ψ, ∇×∇ψ in collocated grid
    #

    #
    # First test
    #
    #Numerical solution
    f1_expr = ∇(ψ)
    v_num = VectorField(local_grid)
    compute!(f1_expr, grid_data, v_num, intx, inty)

    #Analytical solution
    @. v_ana.x.data = fx(local_grid.x, local_grid.y)
    @. v_ana.y.data = fy(local_grid.x, local_grid.y)
    @. v_ana.z.data = 0

    #Compute error
    num_err[n, 1], num_err[n, 2] = compute_error(v_num, v_ana, local_grid, intx, inty)

    #
    # Second test
    #
    #Numerical solution
    f2_expr = ∇²(ψ)
    compute!(f2_expr, grid_data, s_num, intx, inty)

    #Analytical solution
    s_ana = ScalarField(local_grid)
    @. s_ana.field.data = fxx(local_grid.x, local_grid.y) + fyy(local_grid.x, local_grid.y)

    #Compute error
    num_err[n, 3] = compute_error(s_num, s_ana, local_grid, intx, inty)

    #
    # Third test
    #
    #Numerical solution
    f3_expr = ∇ × ∇(ψ)
    v_num = VectorField(local_grid)
    compute!(f3_expr, grid_data, v_num, intx, inty)

    #Analytical solution
    v_ana = VectorField(local_grid) #All zeros

    #Compute error
    num_err[n, 4], num_err[n, 5] = compute_error(v_num, v_ana, local_grid, intx, inty)

    #
    # Fourth test, v_z = psi, calculate curl, then calculate div
    #    
    f4_expr = ∇ × V
    v_num = VectorField(local_grid)
    compute!(f4_expr, grid_data, v_num, intx, inty)

    #Analytical solution
    v_ana = VectorField(local_grid)
    @. v_ana.x.data = fy(local_grid.x, local_grid.y)
    @. v_ana.y.data = -fx(local_grid.x, local_grid.y)

    #Compute error
    num_err[n, 6], num_err[n, 7] = compute_error(v_num, v_ana, local_grid, intx, inty)

    #
    # Calculate div of previous expression
    #
    f5_expr = ∇ ⋅ (f4_expr)
    s_num = ScalarField(local_grid)
    compute!(f5_expr, grid_data, s_num, intx, inty)

    #Analytical solution
    s_ana = ScalarField(local_grid) # All zero from div(curl)

    #Compute error
    num_err[n, 8] = compute_error(s_num, s_ana, local_grid, intx, inty)

    #
    # Same as last test, with staggered grids
    #
    # Eventually these should work also on _some_ of the "ghosts"
    #
    f6_expr = ∇⁺ × V
    v_num = VectorField(local_grid)
    compute!(f6_expr, grid_data, v_num, intx, inty)

    #Analytical solution
    v_ana = VectorField(local_grid)
    @. v_ana.x.data = fy(gridbˣ.x, gridbˣ.y)
    @. v_ana.y.data = -fx(gridbʸ.x, gridbʸ.y)

    num_err[n, 9], num_err[n, 10] = compute_error(v_num, v_ana, local_grid, intx, inty)

    # Divergence of the curl
    f7_expr = ∇⁺ ⋅ (f6_expr)
    s_num = ScalarField(local_grid)
    compute!(f7_expr, grid_data, s_num, intx, inty)

    #Analytical solution
    s_ana = ScalarField(local_grid) # All zero
    num_err[n, 11] = compute_error(s_num, s_ana, local_grid, intx, inty)

    #Curl of the curl (vector Laplacian)
    f8_expr = ∇⁻ × (∇⁺ × (V))
    v_num = VectorField(local_grid)
    compute!(f8_expr, grid_data, v_num, intx, inty)

    #Analytical solution
    v_ana = VectorField(local_grid)
    @. v_ana.z.data = -(fxx(gridvᶻ.x, gridvᶻ.y) + fyy(gridvᶻ.x, gridvᶻ.y))

    nothing, nothing, num_err[n, 12] = compute_error(v_num, v_ana, local_grid, intx, inty)
end

print("Error for unstaggered operations vs number of points\n")
print("∇ψ.x , ∇ψ.y, ∇²ψ, (∇×∇ψ).x, (∇×∇ψ).y\n")
display(num_err)
print("\n\n")

# Let's do a series of test 
# First a simple product scalar times B
#η = 10.0;
#f1 = (η × B);
#r1 = VectorField(grid) # this is the result of the operator applied onto B or whateve#r field in the definition of f
#print("Test η×B\n")
#@btime test!($f1, $r1)

# Then the curl
#f2 = ((∇⁺ × B));
#r2 = VectorField(grid)
#print("Test ∇⁺×B\n")
#@btime test!($f2, $r2)

# Combine 1 and 2 
#f3 = (η × (∇ × B))
#r3 = VectorField(grid)
#print("Test η×∇×B\n")
#@btime test!($f3, $r3)

# Finally something useful...
#f4 = ∇⁻ × (η × (∇⁺ × B))
#r4 = VectorField(grid)
#print("Test ∇⁻×η×∇⁺×B\n")
#@btime test!($f4, $r4)

# We can also compose at will
#f4 = ∇ × (f3)
#r4 = VectorField(grid)
#print("Test composition ∇×(previous calculation)\n")
#@btime test!($f4, $r4)

# Some acceleration...
#f4 = ∇ × (η × (∇ × B))
#r4 = VectorField(grid)
#print("Test composition ∇×(previous calculation), threaded\n")
#@btime test_threads!($f4, $r4)

#let's also do a gradient for fun
#f5 = ∇(B)
#r5 = VectorField(grid)
#print("Test ∇(B)\n")
#@btime test!($f5, $r5)

#f6 = ∇⋅(B)
#r6 = ScalarField(grid)
#print("Test ∇⋅(B)\n")
#@btime test!($f6, $r6)
# the final call to the operator is inline and can be adjusted. It requires data for differentiation.
# This can be dispatched to setup different order of differentation 