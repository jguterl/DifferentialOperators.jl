#Centered differences
# ∂x(v::FieldData, grid_data::AbstractGridDerivatives, i::Index, j::Index) = 0.5 * (v(i + 1, j) - v(i - 1, j)) / grid_data.dx(i, j)
# ∂y(v::FieldData, grid_data::AbstractGridDerivatives, i::Index, j::Index) = 0.5 * (v(i, j + 1) - v(i, j - 1)) / grid_data.dy(i, j)
# ∂z(v::FieldData, grid_data::AbstractGridDerivatives, i::Index, j::Index) = 0

# #Centered differences
# ∂x²(v::FieldData, grid_data::AbstractGridDerivatives, i::Index, j::Index) = (v(i + 1, j) - 2 * v(i, j) + v(i - 1, j)) / grid_data.dx(i, j)^2
# ∂y²(v::FieldData, grid_data::AbstractGridDerivatives, i::Index, j::Index) = (v(i, j + 1) - 2 * v(i, j) + v(i, j - 1)) / grid_data.dy(i, j)^2
# ∂z²(v::FieldData, grid_data::AbstractGridDerivatives, i::Index, j::Index) = 0

# #Forward staggered
# ∂x⁺(v::FieldData, grid_data::AbstractGridDerivatives, i::Index, j::Index) = (v(i + 1, j) - v(i, j)) / grid_data.dx(i, j)
# ∂y⁺(v::FieldData, grid_data::AbstractGridDerivatives, i::Index, j::Index) = (v(i, j + 1) - v(i, j)) / grid_data.dy(i, j)
# ∂z⁺(v::FieldData, grid_data::AbstractGridDerivatives, i::Index, j::Index) = 0

# #Backward staggered
# ∂x⁻(v::FieldData, grid_data::AbstractGridDerivatives, i::Index, j::Index) = (v(i, j) - v(i - 1, j)) / grid_data.dx(i, j)
# ∂y⁻(v::FieldData, grid_data::AbstractGridDerivatives, i::Index, j::Index) = (v(i, j) - v(i, j - 1)) / grid_data.dy(i, j)
# ∂z⁻(v::FieldData, grid_data::AbstractGridDerivatives, i::Index, j::Index) = 0

# This looks like operations for stretched grids -- I'll just replicate for ∂x⁺, ∂x⁻, etc but I'll ask JG
∂x(op::Union{ApplyOperator,FieldData}, grid_data::AbstractGridDerivatives, i::Index, j::Index) = 0.5 * (op(grid_data, i + 1, j) - op(grid_data, i - 1, j)) / grid_data.dx(i, j)
∂y(op::Union{ApplyOperator,FieldData}, grid_data::AbstractGridDerivatives, i::Index, j::Index) = 0.5 * (op(grid_data, i, j + 1) - op(grid_data, i, j - 1)) / grid_data.dy(i, j)
∂z(op::Union{ApplyOperator,FieldData}, grid_data::AbstractGridDerivatives, i::Index, j::Index) = 0.0 #zero_value_array(grid_data, i,j)

∂x⁺(op::Union{ApplyOperator,FieldData}, grid_data::AbstractGridDerivatives, i::Index, j::Index) = (op(grid_data, i + 1, j) - op(grid_data, i, j)) / grid_data.dx(i, j)
∂y⁺(op::Union{ApplyOperator,FieldData}, grid_data::AbstractGridDerivatives, i::Index, j::Index) = (op(grid_data, i, j + 1) - op(grid_data, i, j)) / grid_data.dy(i, j)
∂z⁺(op::Union{ApplyOperator,FieldData}, grid_data::AbstractGridDerivatives, i::Index, j::Index) = 0.0 #zero_value_array(grid_data,i,j)

∂x⁻(op::Union{ApplyOperator,FieldData}, grid_data::AbstractGridDerivatives, i::Index, j::Index) = (op(grid_data, i, j) - op(grid_data, i - 1, j)) / grid_data.dx(i, j)
∂y⁻(op::Union{ApplyOperator,FieldData}, grid_data::AbstractGridDerivatives, i::Index, j::Index) = (op(grid_data, i, j) - op(grid_data, i, j - 1)) / grid_data.dy(i, j)
∂z⁻(op::Union{ApplyOperator,FieldData}, grid_data::AbstractGridDerivatives, i::Index, j::Index) = 0.0 #zero_value_array(grid_data,i,j)


# Fix federico  -> 
#TODO set dx[i,j] for i,j in ghoest cell = dx[boundary] (see grid.jl for that fix)
# ∂x(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.5 * (v(i + 1, j) - v(i - 1, j)) / grid_data.dx(3, 3)
# ∂y(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.5 * (v(i, j + 1) - v(i, j - 1)) / grid_data.dy(3, 3)
# ∂z(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0

# #Centered differences
# ∂x²(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (v(i + 1, j) - 2 * v(i, j) + v(i - 1, j)) / grid_data.dx(3, 3)^2
# ∂y²(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (v(i, j + 1) - 2 * v(i, j) + v(i, j - 1)) / grid_data.dy(3, 3)^2
# ∂z²(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0

# #Forward staggered
# ∂x⁺(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (v(i + 1, j) - v(i, j)) / grid_data.dx(3, 3)
# ∂y⁺(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (v(i, j + 1) - v(i, j)) / grid_data.dy(3, 3)
# ∂z⁺(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0

# #Backward staggered
# ∂x⁻(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (v(i, j) - v(i - 1, j)) / grid_data.dx(3, 3)
# ∂y⁻(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (v(i, j) - v(i, j - 1)) / grid_data.dy(3, 3)
# ∂z⁻(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0

# # This looks like operations for stretched grids -- I'll just replicate for ∂x⁺, ∂x⁻, etc but I'll ask JG
# ∂x(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.5 * (op(grid_data, i + 1, j) - op(grid_data, i - 1, j)) / grid_data.dx(3, 3)
# ∂y(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.5 * (op(grid_data, i, j + 1) - op(grid_data, i, j - 1)) / grid_data.dy(3, 3)
# ∂z(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.0

# ∂x⁺(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (op(grid_data, i + 1, j) - op(grid_data, i, j)) / grid_data.dx(3, 3)
# ∂y⁺(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (op(grid_data, i, j + 1) - op(grid_data, i, j)) / grid_data.dy(3, 3)
# ∂z⁺(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.0

# ∂x⁻(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (op(grid_data, i, j) - op(grid_data, i - 1, j)) / grid_data.dx(3, 3)
# ∂y⁻(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (op(grid_data, i, j) - op(grid_data, i, j - 1)) / grid_data.dy(3, 3)
# ∂z⁻(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.0



struct StencilCoefficients{V,W,Z, dX,dY,dZ} <: AbstractStencilCoefficients{dX,dY,dZ}
    α_x :: V
    α_y :: W
    α_z:: Z
end


@generated_stencil_coefficients_x_y 5

∂x(op::Union{ApplyOperator,FieldData}, grid_data::AbstractGridDerivatives, i::Index, j::Index) = 0.5 * (op(grid_data, i + 1, j) - op(grid_data, i - 1, j)) / grid_data.dx(i, j)
∂x(op::Union{ApplyOperator,FieldData}, stencils::StencilCoefficients{5}, i::Index, j::Index) = sum_coeff_5(op, grid_data, i,j)

sum_coeff_5(op, grid_data, i, j) = α_x[1] + f[i-2] + α_x[2] * f[i-1] + α_x[3] * f[0] + α_x[3] + f[-2] + α_x[4] * f[i+1] + α_x[5] * f[i+2]