#Centered differences
∂x(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.5 * (v(i + 1, j) - v(i - 1, j)) / grid_data.dx(i, j)
∂y(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.5 * (v(i, j + 1) - v(i, j - 1)) / grid_data.dy(i, j)
∂z(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0

#Centered differences
∂x²(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (v(i + 1, j) - 2 * v(i, j) + v(i - 1, j)) / grid_data.dx(i, j)^2
∂y²(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (v(i, j + 1) - 2 * v(i, j) + v(i, j - 1)) / grid_data.dy(i, j)^2
∂z²(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0

#Forward staggered
∂x⁺(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (v(i + 1, j) - v(i, j)) / grid_data.dx(i, j)
∂y⁺(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (v(i, j + 1) - v(i, j)) / grid_data.dy(i, j)
∂z⁺(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0

#Backward staggered
∂x⁻(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (v(i, j) - v(i - 1, j)) / grid_data.dx(i, j)
∂y⁻(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (v(i, j) - v(i, j - 1)) / grid_data.dy(i, j)
∂z⁻(v::FieldData, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0

# This looks like operations for stretched grids -- I'll just replicate for ∂x⁺, ∂x⁻, etc but I'll ask JG
∂x(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.5 * (op(grid_data, i + 1, j) - op(grid_data, i - 1, j)) / grid_data.dx(i, j)
∂y(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.5 * (op(grid_data, i, j + 1) - op(grid_data, i, j - 1)) / grid_data.dy(i, j)
∂z(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.0

∂x⁺(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (op(grid_data, i + 1, j) - op(grid_data, i, j)) / grid_data.dx(i, j)
∂y⁺(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (op(grid_data, i, j + 1) - op(grid_data, i, j)) / grid_data.dy(i, j)
∂z⁺(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.0

∂x⁻(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (op(grid_data, i, j) - op(grid_data, i - 1, j)) / grid_data.dx(i, j)
∂y⁻(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = (op(grid_data, i, j) - op(grid_data, i, j - 1)) / grid_data.dy(i, j)
∂z⁻(op::ApplyOperator, grid_data::AbstractGridDerivatives, i::Int64, j::Int64) = 0.0