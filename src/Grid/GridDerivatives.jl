#
# Schedule this entire class for deletion whenever possible
#
struct CoordSpacings{X<:CoordData,Y<:CoordData,Z<:CoordData,B} <: AbstractCoordSpacings{B}
    dx :: X
    dy :: Y
    dz :: Z
    backend :: B
end

#
# Get rid of this entire class, for now hack to get the correct spacings -- uniform everywhere
#
#function CoordSpacings(coords::Grid, ghost_cells::GhostCells; Kx=1, Ky=1, Kz=1)
function CoordSpacings(coords::LogicalCoords) #, ghost_cells::GhostCells; Kx=1, Ky=1, Kz=1)
    dx = 0*coords.x .+ ( coords.x[2,1] - coords.x[1,1] ) #coords.x .- circshift(coords.x, (1, 0))
    dy = 0*coords.y .+ ( coords.y[1,2].- coords.x[1,1] ) #coords.y .- circshift(coords.y, (0, 1))
    dz = missing
#    set_dx_ghost_cells!(dx, ghost_cells)
#    set_dy_ghost_cells!(dy, ghost_cells)
    CoordSpacings(CoordData(dx), CoordData(dy), CoordData(dz))
end

CoordSpacings(x::AbstractCoordData{B}, y::AbstractCoordData{B}, z::AbstractCoordData{B}) where {B<:Backend} = CoordSpacings(x,y,z,B())

#
# The solution would be for dx to be stored in the grid
#
get_dx(coord_data::CoordSpacings) = CUDA.@allowscalar coord_data.dx.data[3, 3] #TODO: this is wrong if  ng c >2 

get_dy(coord_data::CoordSpacings) = CUDA.@allowscalar coord_data.dy.data[3, 3] # TODO: this is wrong if  ng c >2 

Adapt.@adapt_structure CoordSpacings

export get_dx, get_dy

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
