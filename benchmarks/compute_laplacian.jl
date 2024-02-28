using DifferentialOperators
using LoopVectorization
#set_backend(:cpu)

export compute_laplacian!, compute_laplacian_t!, compute_laplacian_threaded!
#
# Simple laplacian computation to test speed
#
function compute_lapacian!(v, op, StructuredGrid::AbstractStructuredGrid)
    compute_laplacian!(v, op, StructuredGrid.Spacings, StructuredGrid.InteriorPoints[1], StructuredGrid.InteriorPoints[2])
    return nothing
end

function compute_laplacian!(v, op::ScalarField, GridSpacings::AbstractCoordSpacings{B}, i_::UnitRange, j_::UnitRange) where {B<:CPUBackend}
    dli2 = [1. / GridSpacings.dx.data[2,2]^2, 1. / GridSpacings.dy.data[2,2]^2]

    for j in j_
        for i in i_
            v.field.data[i,j] = ( op.field.var.field.data[i+1,j] - 2.0 * op.field.var.field.data[i,j] +  op.field.var.field.data[i-1,j] )  * dli2[1] +
                                ( op.field.var.field.data[i,j+1] - 2.0 * op.field.var.field.data[i,j] +  op.field.var.field.data[i,j-1] ) * dli2[2]
        end
    end
    return nothing
end

#
# Same computation with turbo
#
function compute_lapacian_t!(v, op, StructuredGrid::AbstractStructuredGrid)
    compute_laplacian!(v, op, StructuredGrid.Spacings, StructuredGrid.InteriorPoints[1], StructuredGrid.InteriorPoints[2])
    return nothing
end

function compute_laplacian_t!(v, op::ScalarField, GridSpacings::AbstractCoordSpacings{B}, i_::UnitRange, j_::UnitRange) where {B<:CPUBackend}
    dli2 = [1. / GridSpacings.dx.data[2,2]^2, 1. / GridSpacings.dy.data[2,2]^2]

    @turbo for j in j_
        for i in i_
            v.field.data[i,j] = ( op.field.var.field.data[i+1,j] - 2.0 * op.field.var.field.data[i,j] +  op.field.var.field.data[i-1,j] )  * dli2[1] +
                                ( op.field.var.field.data[i,j+1] - 2.0 * op.field.var.field.data[i,j] +  op.field.var.field.data[i,j-1] ) * dli2[2]
        end
    end
    return nothing
end

#
# Threaded computation with turbo
#
function compute_lapacian_threaded!(v, op, StructuredGrid::AbstractStructuredGrid)
    compute_laplacian_threaded!(v, op, StructuredGrid.Spacings, StructuredGrid.InteriorPoints[1], StructuredGrid.InteriorPoints[2])
    return nothing
end

function compute_laplacian_threaded!(v, op::ScalarField, GridSpacings::AbstractCoordSpacings{B}, i_::UnitRange, j_::UnitRange) where {B<:CPUBackend}
    dxi2 = 1. / GridSpacings.dx.data[2,2]^2
    dyi2 = 1. / GridSpacings.dy.data[2,2]^2

    @tturbo for j in j_
            for i in i_
            v.field.data[i,j] = ( op.field.var.field.data[i+1,j] - 2.0 *op.field.var.field.data[i,j] +  op.field.var.field.data[i-1,j] ) * dxi2 +
                ( op.field.var.field.data[i,j+1] - 2.0 * op.field.var.field.data[i,j] +  op.field.var.field.data[i,j-1] ) * dyi2
#            v.field.data[i,j] = ( op.field.data[i+1,j] - 2.0 *op.field.data[i,j] +  op.field.data[i-1,j] ) * dxi2 +
#                ( op.field.data[i,j+1] - 2.0 * op.field.data[i,j] +  op.field.data[i,j-1] ) * dyi2

#            compute_point!(v, op, GridSpacings, i, j)
        end
    end
    nothing
end
