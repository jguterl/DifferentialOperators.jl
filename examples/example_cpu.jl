using DifferentialOperators
using BenchmarkTools
set_backend!(:cpu)
# A wrapper for testing (f is the operator we are interested in)

# Grid sizes
nx, ny = 100, 100
ig = 2
test!(f, result) = compute!(f, grid_data, result, 1+ig:nx-ig, 1+ig:ny-ig)
test_threads!(f, result) = compute_threads!(f, grid_data, result, 1+ig:nx-ig, 1+ig:ny-ig)

# First we create a grid 
#grid_mhd = MHDGrid(nx, ny; L=[2π, 4π])
coords = LogicalCoords(nx, ny; L=[2π, 4π])

# We also define the data needed to calculated the derivatives (we can define order of accuracy here)
grid_data = GridDerivatives(coords);

dx = grid_data.dx.data[2, 2]
dy = grid_data.dy.data[2, 2]

# define a vector B  with some non-uniform value
kx = 1;
ky = 2;
kz = 3;

ψ = ScalarField(coords);
@. ψ.field = cos(kx * coords.x) * cos(ky * coords.y)


# Contraction product
#M = TensorField(1, 1)
#v = VectorField(1, 1)
#v.x .= 1
#v.y .= 2
#v.z .= 3
#M.xx .= 1
#M.xy .= 2
#M.xz .= 3
#M.yx .= 1*2
#M.yy .= 2*2
#M.yz .= 3*2
#M.zx .= 1*3
#M.zy .= 2*3
#M.zz .= 3*3

#res1 = VectorField(1, 1)
#f! = M ⋅ v
#compute!(f!, grid_data, res1, 1:1, 1:1)

# contraction :
#M = TensorField(1, 1)
#v = VectorField(1, 1)
#v.x .= 1
#v.y .= 2
#v.z .= 3
#M.xx .= 1
#M.xy .= 2
#M.xz .= 3
#M.yx .= 1 * 2
#M.yy .= 2 * 2
#M.yz .= 3 * 2
#M.zx .= 1 * 3
#M.zy .= 2 * 3
#M.zz .= 3 * 3

#res1 = ScalarField(1, 1)
#f! = M ∻ v
#compute!(f!, grid_data, res1, 1:1, 1:1)

#
# ∇ψ in collocated grid
#
print("Test ∇(ψ)\n")
exp1 = ∇(ψ)
res1 = VectorField(coords)
@btime test!($exp1, $res1)

B = VectorField(coords);
@. B.x.data = cos(kx * coords.y) * cos(kx * coords.x);
@. B.y.data = cos(ky * coords.y) * cos(ky * coords.x);
@. B.z.data = cos(kz * coords.y) * cos(kz * coords.x);


# Staggered coordss asuming half a grid point displacement
coords⁺ˣ = LogicalCoords(nx, ny; L=[2π, 4π], d0=[0.5 * dx, 0])
coords⁺ʸ = LogicalCoords(nx, ny; L=[2π, 4π], d0=[0.0, 0.5 * dy])
coords⁺ᶻ = LogicalCoords(nx, ny; L=[2π, 4π], d0=[0.0, 0.0]) #Placehold for z grid


# This vector is defined using a FV like grid, with different location
# for the x,y,z components

B⁺ = VectorField(coords);
@. B⁺.x.data = cos(kx * coords⁺ˣ.y) * cos(kx * coords⁺ˣ.x);
@. B⁺.y.data = cos(ky * coords⁺ʸ.y) * cos(ky * coords⁺ʸ.x);
@. B⁺.z.data = cos(kz * coords⁺ᶻ.y) * cos(kz * coords⁺ᶻ.x);



# Let's do a series of test 
# First a simple product scalar times B
η = 10.0;
f1 = (η × B);
r1 = VectorField(coords) # this is the result of the operator applied onto B or whateve#r field in the definition of f
print("Test η×B\n")
@btime test!($f1, $r1)

# Then the curl
f2 = ((∇⁺ × B));
r2 = VectorField(coords)
print("Test ∇⁺×B\n")
@btime test!($f2, $r2)

# Combine 1 and 2 
f3 = (η × (∇ × B))
r3 = VectorField(coords)
print("Test η×∇×B\n")
@btime test!($f3, $r3)

# Finally something useful...
f4 = ∇⁻ × (η × (∇⁺ × B))
r4 = VectorField(coords)
print("Test ∇⁻×η×∇⁺×B\n")
@btime test!($f4, $r4)

# We can also compose at will
f4 = ∇ × (f3)
r4 = VectorField(coords)
print("Test composition ∇×(previous calculation)\n")
@btime test!($f4, $r4)

# Some acceleration...
f4 = ∇ × (η × (∇ × B))
r4 = VectorField(coords)
print("Test composition ∇×(previous calculation), threaded\n")
@btime test_threads!($f4, $r4)

# let's also do a gradient for fun
#
# This operation does not exist yet
#
#f5 = ∇(B)
#r5 = VectorField(coords)
#print("Test ∇(B)\n")
#@btime test!($f5, $r5)

f6 = ∇ ⋅ (B)
r6 = ScalarField(coords)
print("Test ∇⋅(B)\n")
@btime test!($f6, $r6)
# the final call to the operator is inline and can be adjusted. It requires data for differentiation.
# This can be dispatched to setup different order of differentation 


