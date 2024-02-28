#
# Grid data routines
#

#Centered first derivative
∂x(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = 0.5 * (op(grid_data, i + 1, j) - op(grid_data, i - 1, j)) / grid_data.dx(i, j)
∂y(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = 0.5 * (op(grid_data, i, j + 1) - op(grid_data, i, j - 1)) / grid_data.dy(i, j)
∂z(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = 0.0 

#Centered second derivative
∂x²(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = (op(grid_data, i + 1, j) - 2 * op(grid_data, i, j) + op(grid_data, i - 1, j)) / grid_data.dx(i, j)^2
∂y²(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = (op(grid_data, i, j + 1) - 2 * op(grid_data, i, j) + op(grid_data, i, j - 1)) / grid_data.dy(i, j)^2
∂z²(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = 0.0

#Forward staggered first derivative
∂x⁺(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = (op(grid_data, i + 1, j) - op(grid_data, i, j)) / grid_data.dx(i, j)
∂y⁺(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = (op(grid_data, i, j + 1) - op(grid_data, i, j)) / grid_data.dy(i, j)
∂z⁺(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = 0.0 

#Backward staggered first derivative
∂x⁻(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = (op(grid_data, i, j) - op(grid_data, i - 1, j)) / grid_data.dx(i, j)
∂y⁻(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = (op(grid_data, i, j) - op(grid_data, i, j - 1)) / grid_data.dy(i, j)
∂z⁻(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = 0.0 


#=
#
# dli routines -- these seem to suffer from some type instability although I created a specific class for dli 
# Things that have been tried:
#
# (1) Declare an abstract type for dli with an anonymous function returning the value of dx
# (2) Use a function that returns a fixed dx, dy 
# (3) Use grid_data.dli[ axis ]
#
# These options always end up not working with composition, don't understand why

#Centered first derivative
∂x(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = 0.5 * (op(grid_data, i + 1, j) - op(grid_data, i - 1, j)) * grid_data.dli(1)
∂y(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = 0.5 * (op(grid_data, i, j + 1) - op(grid_data, i, j - 1)) * grid_data.dli(2)
∂z(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = 0.0 

#Centered second derivative
∂x²(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = (op(grid_data, i + 1, j) - 2 * op(grid_data, i, j) + op(grid_data, i - 1, j)) * grid_data.dli2(1)
∂y²(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = (op(grid_data, i, j + 1) - 2 * op(grid_data, i, j) + op(grid_data, i, j - 1)) * grid_data.dli2(2)
∂z²(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = 0.0

#Forward staggered first derivative
∂x⁺(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = (op(grid_data, i + 1, j) - op(grid_data, i, j)) * get_dx(grid_data) #* grid_data.dli(1) #( grid_data.dli[1]
∂y⁺(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = (op(grid_data, i, j + 1) - op(grid_data, i, j)) * get_dy(grid_data) #* grid_data.dli(2)
∂z⁺(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = 0.0 

#Backward staggered first derivative
∂x⁻(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = (op(grid_data, i, j) - op(grid_data, i - 1, j)) * get_dx(grid_data) #grid_data.dli(1)
∂y⁻(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = (op(grid_data, i, j) - op(grid_data, i, j - 1)) * get_dy(grid_data) #grid_data.dli(2)
∂z⁻(op::Union{ApplyOperator,FieldData}, grid_data::AbstractCoordSpacings, i::Index, j::Index) = 0.0 

=#
# struct StencilCoefficients{V,W,Z, dX,dY,dZ} <: AbstractStencilCoefficients{dX,dY,dZ}
#     α_x :: V
#     α_y :: W
#     α_z:: Z
# end

# @generated_stencil_coefficients_x_y 5

# ∂x(op::Union{ApplyOperator,FieldData}, grid_data::AbstractGridDerivatives, i::Index, j::Index) = 0.5 * (op(grid_data, i + 1, j) - op(grid_data, i - 1, j)) / grid_data.dx(i, j)
# ∂x(op::Union{ApplyOperator,FieldData}, stencils::StencilCoefficients{5}, i::Index, j::Index) = sum_coeff_5(op, grid_data, i,j)

# sum_coeff_5(op, grid_data, i, j) = α_x[1] + f[i-2] + α_x[2] * f[i-1] + α_x[3] * f[0] + α_x[3] + f[-2] + α_x[4] * f[i+1] + α_x[5] * f[i+2]

# proposed strategy:
    # used inner iterator for 
