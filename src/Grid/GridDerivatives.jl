#
# Schedule this entire class for deletion whenever possible
#
struct DlData{T,B<:Backend} <: AbstractDlData{B}
    data::T
    backend::B
end
DlData(data) = DlData(current_backend(data), current_backend.value)
Base.size(g::DlData) = size(g.data)
Adapt.@adapt_structure DlData
(d::DlData)(i::Union{Index,Int64,Int32}) = d.data[i]
#(d::DlData)(i::Int64) = d.data[i]


struct CoordSpacings{X<:CoordData,Y<:CoordData,Z<:CoordData,
                    DI<:DlData, DI2<:DlData, B} <: AbstractCoordSpacings{B}
    dx   :: X
    dy   :: Y
    dz   :: Z
    dli  :: DI
    dli2 :: DI2
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
    dli  = [1.0 / dx[1,1]   , 1.0 / dy[1,1]   , 0 ]
    dli2 = [1.0 / dx[1,1]^2 , 1.0 / dy[1,1]^2 , 0 ]
#    set_dx_ghost_cells!(dx, ghost_cells)
#    set_dy_ghost_cells!(dy, ghost_cells)
    CoordSpacings(CoordData(dx), CoordData(dy), CoordData(dz), DlData(dli), DlData(dli2))
end

CoordSpacings(x::AbstractCoordData{B}, y::AbstractCoordData{B}, z::AbstractCoordData{B}, dli::AbstractDlData{B}, dli2::AbstractDlData{B}) where {B<:Backend} = CoordSpacings(x,y,z,dli,dli2,B())

#CoordSpacings(x::AbstractCoordData{B}, y::AbstractCoordData{B}, z::AbstractCoordData{B}, dli::Array{Float64}, dli2::Array{Float64}) where {B<:Backend} = CoordSpacings(x,y,z,dli,dli2,B())
#
# The solution would be for dx to be stored in the grid
#

#get_dx(coord_data::CoordSpacings) = coord_data.dx.data[3, 3] #TODO: this is wrong if  ng c >2 
#get_dy(coord_data::CoordSpacings) = coord_data.dy.data[3, 3] # TODO: this is wrong if  ng c >2 

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
