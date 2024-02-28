export StructuredGrid

struct StructuredGrid{N,C,G,S,I,A} <: AbstractStructuredGrid
    Ndims     :: N
    Coords    :: C
    Nghosts   :: G
    Spacings  :: S
    InteriorPoints :: I
    AllPoints :: A
    
##    Indexes   :: I
##    n         :: N #normal vector
##    e         :: U #unitary vectors
end

function StructuredGrid(npt::Array{Int64}; kw...)
    return StructuredGrid(tuple(npt...); kw...)
end

function StructuredGrid(dims::NTuple{N,Int64}; Nghosts=[0 for i in 1:N], kw...) where N

    npt_tot   = collect(dims) .+ 2*Nghosts
    
    Ndims     = N
    Coords    = LogicalCoords(dims; Nghosts=Nghosts, kw...)
    Nghosts   = Nghosts
    Spacings  = CoordSpacings(Coords)
    InteriorPoints = [ IndexIterator(1,npt_tot[i],Nghosts[i]) for i=1:N ]
    AllPoints      = [ IndexIterator(1,npt_tot[i],         0) for i=1:N ]
    
    #IndexIterators(IndexIterator(1,npt[1],ngc[1]
    #Indexes   = GridIndexes(dims; ngc, kw...)
    #n         = NormalVectors(grid)
    #e         = UnitaryVectors(grid)
    StructuredGrid(Ndims, Coords, Nghosts, Spacings, InteriorPoints, AllPoints) #, n, e)
end

#
# Needs to be a different module / file entirely
# The way it is it entangled the grid with the 
# vector class
#
#struct NormalVectors{SS,NN,WW,EE}
#    S :: SS
#    N :: NN
#    W :: WW
#    E :: EE
#end 

#function NormalVectors(grid::LogicalCoords) 
#    N = VectorField(ones(grid), zeros(grid), zeros(grid))
#    S = VectorField(-ones(grid), zeros(grid), zeros(grid))
#    W = VectorField(zeros(grid), -ones(grid), zeros(grid))
#    E = VectorField(zeros(grid), ones(grid), zeros(grid))
#    return NormalVectors(N,S,W,E)
#end

#
# All this needs to be a different struct / file entirely
#
#UnitaryVectors(grid::LogicalCoords) = VectorField(VectorField(ones(grid), zeros(grid),zeros(grid)),VectorField(zeros(grid), ones(grid),zeros(grid)),VectorField(zeros(grid), zeros(grid),ones(grid)))

