export StructuredGrid

struct StructuredGrid{C,G,D,I} <: AbstractStructuredGrid
    Coords    :: C
    Ghosts    :: G
    grid_data :: D
    indexes   :: I
#    n         :: N #normal vector
#    e         :: U #unitary vectors
end

function StructuredGrid(npt::Array{Int64}; kw...)
    return StructuredGrid(tuple(npt...); kw...)
end

function StructuredGrid(dims::NTuple{N,Int64}; kw...) where N
    Coords    = LogicalCoords(dims; kw...) 
    grid_data = GridDerivatives(grid)
    indexes   = GridIndexes(dims; kw...)

    #grid_data = GridDerivatives(grid, indexes.ghost_cells)
    #n         = NormalVectors(grid)
    #e         = UnitaryVectors(grid)
    StructuredGrid(grid, grid_data, indexes) #, n, e)
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

