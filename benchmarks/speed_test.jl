using DifferentialOperators
using BenchmarkTools
using PyPlot
set_backend!(:cpu)

include("compute_laplacian.jl")

# A wrapper for testing (f is the operator we are interested in)

# Grid sizes
nx, ny = 32, 32
ig = 1
L = [2π,4π]
nreps = 8
ng = [ig,ig]
nmethods = 3
timings = zeros(nmethods,nreps)

BenchmarkTools.DEFAULT_PARAMETERS.samples=1000
tnorm = 1e9
for n = 1:nreps
    # First we create a grid 
    #grid_mhd = MHDGrid(nx, ny; L=[2π, 4π])
    npt    = [nx*2^(n-1), ny*2^(n-1)]
    coords = LogicalCoords(npt; L=[2π, 4π], Nghosts=ng )

    ixr = (ng[1]+1):(npt[1]+ng[1])
    iyr = (ng[2]+1):(npt[2]+ng[2])

    # We also define the data needed to calculated the derivatives (we can define order of accuracy here)
    grid_data = CoordSpacings(coords);

    # define a vector B  with some non-uniform value
    kx = 1;
    ky = 2;
    kz = 3;

    ψ = ScalarField(coords);
    @. ψ.field = cos(kx * coords.x) * cos(ky * coords.y)

    res0 = ScalarField(coords)
    exp0 = ∇²(ψ)
    res00 = ScalarField(coords)
    
    #
    # Compare against a simple FD written in loop form as sanity check
    #

    print("\nTest ∇^2(ψ) with composed ops\n")
    b = @benchmark compute!($res0, $exp0, $grid_data, $ixr, $iyr)
    timings[1,n] = mean(b.times) / tnorm


    
    print("Test ∇^2(ψ) with hard coded FDs\n")
    b = @benchmark compute_laplacian!($res0, $exp0, $grid_data, $ixr, $iyr)
    timings[2,n] = mean(b.times) / tnorm
    
    
    print("Test ∇^2(ψ) with @turbo\n")
    b = @benchmark compute_laplacian_t!($res00, $exp0, $grid_data, $ixr, $iyr)
    timings[3,n] = mean(b.times) / tnorm
    
    #=
    print("Test ∇^2(ψ) with @tturbo\n")
    b = @benchmark compute_laplacian_threaded!($res00, $exp0, $grid_data, $ixr, $iyr)
    timings[4,n] = mean(b.times) / tnorm
    =#
end

#print(timings)
timings

time_gfortran_O3 = [6.6399946808815008E-007 3.3589992672204970E-006 1.1514999903738499E-005 4.4792000204324724E-005 2.0844700001180171E-004 1.1071009999141098E-003 4.3027710001915694E-003 1.7517017000354827E-002  ]

time_gfortran_O0 = [6.6900001838803288E-006 2.8371999971568586E-005 1.1224899999797344E-004 4.4935400038957597E-004 1.8203399991616608E-003 7.3568259999156000E-003 2.9822660000063478E-002 0.11842941599991172]

time_octave = [2.9371e-05   1.0258e-04   3.8070e-04   1.5767e-03   6.0199e-03   2.4721e-02   9.9547e-02   4.5187e-01]

ttimes = [timings; time_gfortran_O3; time_gfortran_O0; time_octave]'
npt=[32*2^(i-1) for i=1:8]
loglog(npt,ttimes)
xlabel("Number of points")
ylabel("Time [s]")
legend(["DifferentialOperators", "Julia FD", "Julia FD + @turbo", "gfortran -O3", "gfortran -O0", "Octave sparse"])
