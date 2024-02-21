using CUDA
using Adapt

abstract type Backend end
struct CPUBackend <: Backend end
struct CUDABackend <: Backend end


(b::CUDABackend)(t) = t
(b::CPUBackend)(t) = t
(b::CUDABackend)(t::Array) = CUDA.CuArray(t)

available_backends = [:serial, :cpu, :gpu, :cuda]
function Backend(backend::Symbol)
    backend ∈ [:cpu, :serial] && return CPUBackend()
    backend ∈ [:gpu, :cuda] && return CUDABackend()
    error("`$backend` unknown. Available backends: $available_backends")
end

mutable struct CurrentBackend
    value :: Backend
end

const current_backend = CurrentBackend(CPUBackend())
(cbk::CurrentBackend)(args... ;kw...) = cbk.value(args...; kw...)

get_current_backend = () -> current_backend
function set_backend!(backend::Symbol)
    @assert backend ∈ available_backends "`$backend` unknown. Available backends: $available_backends"
    current_backend.value = Backend(backend)
    check_backend!(current_backend)
    println("Backend set to $(current_backend.value)")
end
check_backend!(current_backend::CurrentBackend) = check_backend!(current_backend.value)
check_backend!(::CPUBackend) = nothing
check_backend!(::CUDABackend) = !CUDA.functional() && error("CUDA is not functional...")


#     cuda_copy(v::Array{Symbol,D}, ::CUDARun) where {D} = CUDA.CuVector(Vector{Missing}())
#     cuda_copy(v::CUDA.CuArray, ::CUDARun) = v
#     cuda_copy(x::AbstractArray{T}, ::CUDABuilder) where {T<:Number} = CuArray(x)
#     cuda_copy(x::AbstractArray{T}, ::CUDARun) where {T<:Number} = CuArray(x)
#     function cuda_zeros(type, dims, v)::CuArray
#         arr = CUDA.zeros(type, dims...)
#         CUDA.fill!(arr, v)
#         return arr
#     end
#     function cuda_zeros(neq::Int64)::CuArray
#         arr = CUDA.zeros(Float64, neq)
#         return arr
#     end
# else
#     cuda_copy(x::AbstractArray{T}, ::CUDARun) where {T<:Number} = x
#     cuda_copy(v::Array{Symbol,D}, ::CUDARun) where {D} = Vector{Missing}()
#     function cuda_copy(x::AbstractArray{T}, ::CUDABuilder) where {T<:Number}
#         x
#     end
#     function cuda_zeros(type, dims, fill)::Array
#         if applicable(zero, type)
#             arr = zeros(type, dims...)
#             fill!(arr, convert(type, fill))
#         else
#             arr = Array{type}(undef, args...)
#             fill!(arr, fill)
#         end
#         return arr
#     end
#     function cuda_zeros(neq::Int64)
#         arr = zeros(neq)
#         return arr
#     end

# end
# function cuda_copy(v::Array{Symbol,D}, ::CUDABuilder) where {D}
#     v
# end
# backend = Backend{}
# hardware = 
# function set_hardware!(s::Symbol)
