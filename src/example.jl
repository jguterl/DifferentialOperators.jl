using DifferentialOperators

# First we create a grid 
grid = Grid(10, 10; L=[10.0, 2.0])
@show grid

# We also define the data needed to calculated the derivatives (we can define order of accuracy here)
grid_data = GridDerivatives(grid);

# define a vector B  with some non-uniform value
B = VectorField(grid)
@. B.y.data = exp(-grid.y - grid.x)
@. B.x.data = exp(-grid.y - grid.x)
@. B.z.data = exp(-grid.y - grid.x)

nx,ny = size(grid)
#a wrapper for testing (f is the operator we are interested in)
test!(f::F, result) where {F<:VectorField} = compute!(f, grid_data, result, 3:nx-3, 3:ny-2)

#Let's do a series of test 
# First a simple product scalar times B
η = 10.0
f1 = (η × B)
r1 = VectorField(grid) # this is the result of the operator applied onto B or whatever field in the definition of f
@btime test!($f1, $r1)

# Thne the curl
f2 = ((∇ × B))
r2 = VectorField(grid)
@btime test!($f2, $r2)

# Combine 1 and 2 
f3 = (η × (∇ × B))
r3 = VectorField(grid)
@btime test!($f3, $r3)

# Finally something useful...
f4 = ∇ × (η × (∇ × B))
r4 = VectorField(grid)
@btime test!($f4, $r4)

#let's also do a gradient for fun
f5 = ∇(B)
r5 = VectorField(grid)
@btime test!($f5, $r5)
# the final call to the operator is inline and can be adjusted. It requires data for differentiation.
# This can be dispatched to setup different order of differentation 


