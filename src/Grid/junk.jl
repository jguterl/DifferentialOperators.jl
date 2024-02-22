export MHDGrid

const Grid1D{X}     = LogicalCoords{X,Missing,Missing}
const Grid2D{X,Y}   = LogicalCoords{X,Y,Missing}
const Grid3D{X,Y,Z} = LogicalCoords{X,Y,Z}

#
# Needs to be a different module / file entirely
#
struct NormalVectors{SS,NN,WW,EE}
    S :: SS
    N :: NN
    W :: WW
    E :: EE
end 

function NormalVectors(grid::LogicalCoords) 
    N = VectorField(ones(grid), zeros(grid), zeros(grid))
    S = VectorField(-ones(grid), zeros(grid), zeros(grid))
    W = VectorField(zeros(grid), -ones(grid), zeros(grid))
    E = VectorField(zeros(grid), ones(grid), zeros(grid))
    return NormalVectors(N,S,W,E)
end

UnitaryVectors(grid::LogicalCoords) = VectorField(VectorField(ones(grid), zeros(grid),zeros(grid)),VectorField(zeros(grid), ones(grid),zeros(grid)),VectorField(zeros(grid), zeros(grid),ones(grid)))

#
# All this needs to be a different struct / file entirely
#
struct MHDGrid{G,D,I,N,U} <: AbstractMHDGrid
    grid      :: G
    grid_data :: D
    indexes   :: I
    n         :: N #normal vector
    e         :: U #unitary vectors
end

function MHDGrid(nx::Int64, ny::Int64; kw...)
    return MHDGrid((nx,ny); kw...)
end

function MHDGrid(dims::NTuple{N,Int64}; kw...) where N
    grid      = LogicalCoords(dims; kw...) 
    indexes   = GridIndexes(dims; kw...)
    grid_data = GridDerivatives(grid, indexes.ghost_cells)
    n         = NormalVectors(grid)
    e         = UnitaryVectors(grid)
    MHDGrid(grid, grid_data, indexes, n, e)
end


# TODO: FH check that
#function set_dx_ghost_cells!(dx, ghost_cells::GhostCells) 
#    ghost_cells.nx == 0 && return 
#    dx[1:1+ghost_cells.nx-1,:]  = dx[1+ghost_cells.nx,:] 
#    dx[end-ghost_cells.nx+1:end,:]  = dx[end-ghost_cells.nx,:]
#end

# TODO: FH check that
#function set_dy_ghost_cells!(dy, ghost_cells::GhostCells)
#    ghost_cells.ny == 0 && return
#    dy[:,1:1+ghost_cells.ny-1] = dy[:,1+ghost_cells.ny]
#    dy[:,end-ghost_cells.ny+1:end] = dy[:,end-ghost_cells.ny]
#end

#export IndexIterator

# -- junk ----
# derivative operators
# abstract type DerivativeOperator{K} end
# struct dX{K} <: DerivativeOperator{K} end
# struct dY{K} <: DerivativeOperator{K} end
# struct dZ{K} <: DerivativeOperator{K} end

# struct ApplyDerivativeOperator{T<:DerivativeOperator,X,Y,Z}
#     dx::X
#     dy::Y
#     dz::Z
# end

# ApplyDerivativeOperator{T}(x::X, y::Y, z::Z) where {T,X,Y,Z} = ApplyDerivativeOperator{T,X,Y,Z}(x, y, z)
# function ApplyDerivativeOperator{T}(grid::Grid) where {T<:DerivativeOperator}
#     dx = grid.x .- circshift(grid.x, (1, 0))
#     dy = grid.y .- circshift(grid.y, (0, 1))
#     dz = missing
#     ApplyDerivativeOperator{T}(GridData(dx), GridData(dy), GridData(dz))
# end
