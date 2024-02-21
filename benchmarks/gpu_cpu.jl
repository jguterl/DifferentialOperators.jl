using DifferentialOperators
using CUDA
using BenchmarkTools

kx = 1;
ky = 2;
kz = 3;
η = 10.0;


# Grid sizes
nx, ny = 1000, 1000


set_backend!(:cuda)
grid_mhd = MHDGrid(nx, ny; L=[2π, 4π])
grid = grid_mhd.grid
grid_data = GridDerivatives(grid);
dx = get_dx(grid_data)
dy = get_dy(grid_data)

B = VectorField(grid);
@. B.x.data = cos(kx * grid.y) * cos(kx * grid.x);
@. B.y.data = cos(ky * grid.y) * cos(ky * grid.x);
@. B.z.data = cos(kz * grid.y) * cos(kz * grid.x);

ψ = ScalarField(grid);
@. ψ.field = cos(kx * grid.x) * cos(ky * grid.y)


test_gpu!(f, result) = CUDA.@sync @cuda compute!(f, grid_data, result, IndexIterator(grid_mhd.indexes.inner_iter.i), IndexIterator(grid_mhd.indexes.inner_iter.j))


print("Test ∇(ψ)\n")
exp1 = ∇(ψ)
res1 = VectorField(grid)
test1_gpu = @btime test_gpu!(exp1, res1)

f4 = ∇⁻ × (η × (∇⁺ × B))
r4 = VectorField(grid)
print("Test ∇⁻×η×∇⁺×B\n")
test4_gpu = @btime test_gpu!($f4, $r4)

# -------------- cpu ----------------
set_backend!(:cpu)
grid_mhd = MHDGrid(nx, ny; L=[2π, 4π])
grid = grid_mhd.grid
grid_data = GridDerivatives(grid);
dx = get_dx(grid_data)
dy = get_dy(grid_data)

B = VectorField(grid);
@. B.x.data = cos(kx * grid.y) * cos(kx * grid.x);
@. B.y.data = cos(ky * grid.y) * cos(ky * grid.x);
@. B.z.data = cos(kz * grid.y) * cos(kz * grid.x);

ψ = ScalarField(grid);
@. ψ.field = cos(kx * grid.x) * cos(ky * grid.y)


test_cpu!(f, result) = compute!(f, grid_data, result, IndexIterator(grid_mhd.indexes.inner_iter.i), IndexIterator(grid_mhd.indexes.inner_iter.j))
test_threads!(f, result) = compute_threads!(f, grid_data, result, IndexIterator(grid_mhd.indexes.inner_iter.i), IndexIterator(grid_mhd.indexes.inner_iter.j))

print("Test ∇(ψ)\n")
exp1 = ∇(ψ)
res1 = VectorField(grid)
test1_cpu = @btime test_cpu!(exp1, res1)

f4 = ∇⁻ × (η × (∇⁺ × B))
r4 = VectorField(grid)
print("Test ∇⁻×η×∇⁺×B\n")
test4_cpu = @btime test_cpu!($f4, $r4)

println("Test ∇(ψ) threads")
exp1 = ∇(ψ)
res1 = VectorField(grid)
test1_threads = @btime test_threads!(exp1, res1)

f4 = ∇⁻ × (η × (∇⁺ × B))
r4 = VectorField(grid)
println("Test ∇⁻×η×∇⁺×B threads ")
test4_threads = @btime test_threads!($f4, $r4)

println("--- Summary ---")
println("Test 1 : cpu - $test1_cpu")