using DifferentialOperators
using BenchmarkTools
using Symbolics
import LinearAlgebra.norm as norm


function compute_error(f1::ScalarField, f0::ScalarField, g::LogicalCoords, intx, inty)
    df = ScalarField(g)
    @. df.field.data = f1.field.data - f0.field.data
    return norm(df.field.data[intx, inty], Inf)
end

function compute_error(v1::VectorField, v0::VectorField, g::LogicalCoords, intx, inty)
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
ng  = [1, 1]
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

    npts = [nx, ny]
    #ng   = [ 1,  1]
    L    = [Lx, Ly]
    
    # First we create a grid that start at x=0 on the first interior point
    full_grid = StructuredGrid( npts; L=L, Nghosts=ng, d0=[0.0, 0.0] )
    local_coords = full_grid.Coords
#    local_coords = LogicalCoords( npts ; L=[Lx, Ly], d0=[0.0, 0.0], Nghosts=[1, 1])

    # We also define the data needed to calculated the derivatives (we can define order of accuracy here)
    grid_data = full_grid.Spacings
#    grid_data = CoordSpacings(local_coords)

    intx = full_grid.InteriorPoints[1] #1+ng:nx+ng
    inty = full_grid.InteriorPoints[2] #1+ng:ny+ng

#    intx2 = 1+ng2:nx+ng2
#    inty2 = 1+ng2:ny+ng2

    dx = grid_data.dx.data[2, 2]
    dy = grid_data.dy.data[2, 2]

    # Staggered grids asuming half a grid point displacement
    # This is the standard finite volume grid for fluxes through faces
    gridvˣ = LogicalCoords( npts ; L=[Lx, Ly], d0=[0.5 * dx, 0.0 * dy], Nghosts=[1, 1])
    gridvʸ = LogicalCoords( npts ; L=[Lx, Ly], d0=[0.0 * dx, 0.5 * dy], Nghosts=[1, 1])
    gridvᶻ = LogicalCoords( npts ; L=[Lx, Ly], d0=[0.0 * dx, 0.0 * dy], Nghosts=[1, 1]) #Placehold for z grid

    # This is the adjoint grid produced using curl(v) 
    coordsbˣ = LogicalCoords( npts ; L=[Lx, Ly], d0=[0.0 * dx, 0.5 * dy], Nghosts=[1, 1]) # 0.5*dz
    coordsbʸ = LogicalCoords( npts ; L=[Lx, Ly], d0=[0.5 * dx, 0.0 * dy], Nghosts=[1, 1]) # 0.5*dz
    coordsbᶻ = LogicalCoords( npts ; L=[Lx, Ly], d0=[0.5 * dx, 0.5 * dy], Nghosts=[1, 1]) #Placehold for z grid

    ψ = ScalarField(local_coords)
    @. ψ.field.data = cos(kx * local_coords.x) * cos(ky * local_coords.y)

    V = VectorField(local_coords)
    @. V.x.data = 0.0
    @. V.y.data = 0.0
    @. V.z.data = ψ.field.data

    v_num = VectorField(local_coords)
    s_num = ScalarField(local_coords)
    v_ana = VectorField(local_coords)
    s_ana = ScalarField(local_coords)

    #
    # ∇ψ, ∇²ψ, ∇×∇ψ in collocated grid
    #

    #
    # First test
    #
    #Numerical solution
    f1_expr = ∇(ψ)
    v_num = VectorField(local_coords)
    compute!(v_num, f1_expr, full_grid)
#    compute!(v_num, f1_expr, grid_data, intx, inty)

    #Analytical solution
    @. v_ana.x.data = fx(local_coords.x, local_coords.y)
    @. v_ana.y.data = fy(local_coords.x, local_coords.y)
    @. v_ana.z.data = 0

    #Compute error
    num_err[n, 1], num_err[n, 2] = compute_error(v_num, v_ana, local_coords, intx, inty)

    #
    # Second test
    #
    #Numerical solution
    f2_expr = ∇²(ψ)
    compute!(s_num, f2_expr, full_grid)
    #compute!(s_num, f2_expr, grid_data, intx, inty)

    #Analytical solution
    s_ana = ScalarField(local_coords)
    @. s_ana.field.data = fxx(local_coords.x, local_coords.y) + fyy(local_coords.x, local_coords.y)

    #Compute error
    num_err[n, 3] = compute_error(s_num, s_ana, local_coords, intx, inty)

    #
    # Third test
    #
    #Numerical solution
    f3_expr = ∇ × ∇(ψ)
    v_num = VectorField(local_coords)
    compute!(v_num, f3_expr, full_grid)
    #compute!(v_num, f3_expr, grid_data, intx, inty)

    #Analytical solution
    v_ana = VectorField(local_coords) #All zeros

    #Compute error
    num_err[n, 4], num_err[n, 5] = compute_error(v_num, v_ana, local_coords, intx, inty)

    #
    # Fourth test, v_z = psi, calculate curl, then calculate div
    #    
    f4_expr = ∇ × V
    v_num = VectorField(local_coords)
    compute!(v_num, f4_expr, full_grid)
#    compute!(v_num, f4_expr, grid_data, intx, inty)

    #Analytical solution
    v_ana = VectorField(local_coords)
    @. v_ana.x.data = fy(local_coords.x, local_coords.y)
    @. v_ana.y.data = -fx(local_coords.x, local_coords.y)

    #Compute error
    num_err[n, 6], num_err[n, 7] = compute_error(v_num, v_ana, local_coords, intx, inty)

    #
    # Calculate div of previous expression
    #
    f5_expr = ∇ ⋅ (f4_expr)
    s_num = ScalarField(local_coords)
    compute!(s_num, f5_expr, full_grid)
#    compute!(s_num, f5_expr, grid_data, intx, inty)

    #Analytical solution
    s_ana = ScalarField(local_coords) # All zero from div(curl)

    #Compute error
    num_err[n, 8] = compute_error(s_num, s_ana, local_coords, intx, inty)

    #
    # Same as last test, with staggered grids
    #
    # Eventually these should work also on _some_ of the "ghosts"
    #
    f6_expr = ∇⁺ × V
    #    f6_expr = ∇⁺(ψ)
    
    v_num = VectorField(local_coords)
    compute!(v_num, f6_expr, full_grid)
#    compute!(v_num, f6_expr, grid_data, intx, inty)

    #Analytical solution
    v_ana = VectorField(local_coords)
#    @. v_ana.x.data = fx(gridvˣ.x, gridvˣ.y)
#    @. v_ana.y.data = fy(gridvʸ.x, gridvʸ.y)

    @. v_ana.x.data = +fy(coordsbˣ.x, coordsbˣ.y)
    @. v_ana.y.data = -fx(coordsbʸ.x, coordsbʸ.y)

    num_err[n, 9], num_err[n, 10] = compute_error(v_num, v_ana, local_coords, intx, inty)

    # Divergence of the curl
    f7_expr = ∇⁺ ⋅ (f6_expr)
    s_num = ScalarField(local_coords)
    compute!(s_num, f7_expr, full_grid)
#    compute!(s_num, f7_expr, grid_data, intx, inty)

    #Analytical solution
    s_ana = ScalarField(local_coords) # All zero
    num_err[n, 11] = compute_error(s_num, s_ana, local_coords, intx, inty)

    #Curl of the curl (vector Laplacian)
    f8_expr = ∇⁻ × (∇⁺ × (V))
    v_num = VectorField(local_coords)
    compute!(v_num, f8_expr, full_grid)
#    compute!(v_num, f8_expr, grid_data, intx, inty)

    #Analytical solution
    v_ana = VectorField(local_coords)
    @. v_ana.z.data = -(fxx(gridvᶻ.x, gridvᶻ.y) + fyy(gridvᶻ.x, gridvᶻ.y))

    nothing, nothing, num_err[n, 12] = compute_error(v_num, v_ana, local_coords, intx, inty)
end

print("Error for unstaggered operations vs number of points\n")
print("∇ψ.x , ∇ψ.y, ∇²ψ, (∇×∇ψ).x, (∇×∇ψ).y\n")
display(num_err)
print("\n\n")
