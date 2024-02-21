
# ---- time operator-----
#
# This should not be here, and it's also likely not even needed
#
abstract type Apply∂ₜ{C} end 

function Apply∂ₜ{C}(Δt::Float64, v :: Union{Field,ApplyOperator}, order) where {C}
    order == 1 && return Apply∂ₜ1order{C}(Δt, v)
    error("not implemented yet....")
    order == 2 && return Apply∂ₜ2order{C}(Δt, v)
end

∂ₜ(v::T, Δt::Float64, order=1) where {T<:Union{Field,ApplyOperator}} = get_base_type(T)((Apply∂ₜ{fn}(Δt, v, order) for fn in fieldnames(T))...)

struct Apply∂ₜ1order{C,F<:Vector,T<:Union{Field,ApplyOperator}} <: Apply∂ₜ{C}
    Δt::F
    var::T
    var_old::T
end

Apply∂ₜ1order{C}(Δt::Float64, v::T) where {C,T} = Apply∂ₜ1order{C,Vector{Float64},T}([Δt], v,copy(v))

struct Apply∂ₜ2order{C,F<:Vector,T<:Union{Field,ApplyOperator}} <: Apply∂ₜ{C}# let's not use mutable struct because of gpu
    Δt::F 
    var::T
    var_old::T
    var_old2::T
end

Apply∂ₜ2order{C}(Δt::Float64, v::T) where {C,T} = Apply∂ₜ2order{C,Vector{Float64},T}([Δt, Δt], v, copy(v), copy(v))

set_dt!(∂ₜ::Apply∂ₜ1order, dt::Float64) = ∂ₜ.Δt[1] = dt

(∂ₜ::Apply∂ₜ1order{:x,F,T})(args...) where {F,T}= (∂ₜ.var.x(args...) - ∂ₜ.var_old.x(args...)) / Δt[1]
(∂ₜ::Apply∂ₜ1order{:y,F,T})(args...) where {F, T} = (∂ₜ.var.y(args...) - ∂ₜ.var_old.y(args...)) / Δt[1]
(∂ₜ::Apply∂ₜ1order{:z,F,T})(args...) where {F, T} = (∂ₜ.var.z(args...) - ∂ₜ.var_old.z(args...)) / Δt[1]
(∂ₜ::Apply∂ₜ1order{:field,F,T})(args...) where {F, T} = (∂ₜ.var.field(args...) - ∂ₜ.var_old.field(args...)) / Δt[1]

(∂ₜ::Apply∂ₜ2order)(args...) = error() # TODO .... 

export ∂ₜ
