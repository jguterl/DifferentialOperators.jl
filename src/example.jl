using DifferentialOperators
using Symbolics
using BenchmarkTools

import LinearAlgebra.norm as norm


function compute_error(f1::ScalarField, f0::ScalarField, g::Grid, intx, inty)
    df = ScalarField(g)
    @. df.field.data = f1.field.data - f0.field.data 
    return norm(df.field.data[intx,inty],Inf)
end

function compute_error(v1::VectorField, v0::VectorField, g::Grid, intx, inty)
    dv = VectorField(g)
    @. dv.x.data = v1.x.data - v0.x.data
    @. dv.y.data = v1.y.data - v0.y.data 
    return norm(dv.x.data[intx,inty],Inf), norm(dv.y.data[intx,inty],Inf)
end

# A wrapper for testing (f is the operator we are interested in)
#test!(f, result, intx, inty)          = compute!(f, grid_data, result, intx, inty)

#test_threads!(f, result, intx, inty)  = compute_threads!(f, grid_data, result, intx, inty)

# Analytical solutions to the problems
kx = 1; ky = 2; kz = 3;

@variables x y
∂x     = Differential(x)
∂y     = Differential(y)

ψ₀     = cos(kx*x) * cos(ky*y)
∂x_ψ₀  = expand_derivatives(∂x(ψ₀))
∂y_ψ₀  = expand_derivatives(∂y(ψ₀))
∂xx_ψ₀ = expand_derivatives(∂x(∂x(ψ₀)))
∂yy_ψ₀ = expand_derivatives(∂y(∂y(ψ₀)))

fx  = eval( build_function( ∂x_ψ₀ , x , y ) )
fy  = eval( build_function( ∂y_ψ₀ , x , y ) )
fxx = eval( build_function( ∂xx_ψ₀, x , y ) )
fyy = eval( build_function( ∂yy_ψ₀, x , y ) )

# Grid sizes
nx0, ny0 = 32, 32
ng       = 1
nreps    = 6

nscltests = 1
nvectests = 2
ndims     = 2
nerrs     = ndims * nvectests + nscltests
err       = zeros(nerrs,nreps)

for n=1:nreps

    nx = nx0 * 2^(n-1)
    ny = ny0 * 2^(n-1)
    
    intx = 1+ng:nx-ng
    inty = 1+ng:ny-ng

    # First we create a grid 
    grid = Grid(nx, ny; L=[2π, 4π])

    # We also define the data needed to calculated the derivatives (we can define order of accuracy here)
    grid_data = GridDerivatives(grid);

    dx = grid_data.dx.data[2,2]
    dy = grid_data.dy.data[2,2]

    ψ = ScalarField(grid);
    @. ψ.field.data = cos(kx*grid.x) * cos(ky*grid.y)

    #
    # ∇ψ, ∇²ψ, ∇×∇ψ in collocated grid
    #

    #
    # First test
    #
    #print("Test ∇(ψ)\n")
    #Analytical solution
    f1_ana = VectorField(grid)
    @. f1_ana.x.data = fx( grid.x, grid.y )
    @. f1_ana.y.data = fy( grid.x, grid.y )

    #Numerical solution
    f1_expr = ∇(ψ)
    f1_num  = VectorField(grid)
    compute!(f1_expr, grid_data, f1_num, intx, inty)

    #Error calculation
    err[1,n],err[2,n] = compute_error( f1_num, f1_ana, grid, intx, inty )
    #print("Norm inf=", f1_err_x, " ", f1_err_y, "\n\n")

    #
    # Second test
    #
    #print("Test ∇²(ψ)\n")
    #Analytical solution
    f2_ana = ScalarField(grid)
    @. f2_ana.field.data = fxx(grid.x, grid.y) + fyy(grid.x, grid.y)

    #Numerical solution
    f2_expr = ∇²(ψ)
    f2_num  = ScalarField(grid)
    compute!(f2_expr, grid_data, f2_num, intx, inty)

    #Residual
    err[3,n] = compute_error( f2_num, f2_ana, grid, intx, inty )
    #print("Norm inf=", f2_err, "\n\n")

    #
    # Third test
    #
    #print("Test ∇×∇(ψ)\n")
    #Analytical solution
    f3_ana = VectorField(grid) #All zeros

    #Numerical solution
    f3_expr = ∇×∇(ψ)
    f3_num  = VectorField(grid)
    compute!(f3_expr, grid_data, f3_num, intx, inty)
    err[4,n],err[5,n] = compute_error( f3_num, f3_ana, grid, intx, inty )
    #print("Norm inf=", f3_err_x, " ", f3_err_y, "\n\n")

end

#B = VectorField(grid);
#@. B.x.data = cos(kx*grid.y) * cos(kx*grid.x);
#@. B.y.data = cos(ky*grid.y) * cos(ky*grid.x);
#@. B.z.data = cos(kz*grid.y) * cos(kz*grid.x);


# Staggered grids asuming half a grid point displacement
#grid⁺ˣ = Grid(nx, ny; L=[2π, 4π], d0=[0.5*dx, 0     ])
#grid⁺ʸ = Grid(nx, ny; L=[2π, 4π], d0=[   0.0, 0.5*dy])
#grid⁺ᶻ = Grid(nx, ny; L=[2π, 4π], d0=[   0.0, 0.0   ]) #Placehold for z grid

#
# This vector is defined using a FV like grid, with different location
# for the x,y,z components
#
#B⁺ = VectorField(grid);
#@. B⁺.x.data = cos(kx*grid⁺ˣ.y) * cos(kx*grid⁺ˣ.x);
#@. B⁺.y.data = cos(ky*grid⁺ʸ.y) * cos(ky*grid⁺ʸ.x);
#@. B⁺.z.data = cos(kz*grid⁺ᶻ.y) * cos(kz*grid⁺ᶻ.x);



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


