# module StaggeredKernels

export Tensor, AbstractTensor, #=Vector,=# L2, J1, J2, J3, I1, I2, I3, tr, dev, divergence, grad, grad2, symgrad, curl, diag, TensorOp

abstract type AbstractTensor end

abstract type TensorSymmetry{O} end

struct Tensor{S<:TensorSymmetry, T <: NamedTuple} <: AbstractTensor
	cpnts::T
end

struct TensorProd{T1 <: AbstractTensor, T2 <: AbstractTensor} <: AbstractTensor
	t1::T1
	t2::T2
end

struct TensorExpr{T <: Tuple} <: AbstractTensor
	contents::T
end

(TensorExpr(args...)) = TensorExpr(tuple(args...))
TensorOp(op::Symbol, args...) = TensorExpr(Val(:call), Val(op), args...)

(diag(x::Tensor{S}, f) where {S}) = 
	Tensor((; zip(keys(x.cpnts), diag(getproperty(x, c), getproperty(f, c)) for c in keys(x.cpnts))...), S)
(diag(x::Tensor{S}, f, offset) where {S}) = 
	Tensor((; zip(keys(x.cpnts), diag(getproperty(x, c), getproperty(f, c), offset) for c in keys(x.cpnts))...), S)

(diag(x::Tensor{S}, f::Tuple) where {S}) = 
	Tensor((; zip(keys(x.cpnts), diag(getproperty(x, c), getproperty.(f, c)) for c in keys(x.cpnts))...), S)
(diag(x::Tensor{S}, f::Tuple, offset) where {S}) = 
	Tensor((; zip(keys(x.cpnts), diag(getproperty(x, c), getproperty.(f, c), offset) for c in keys(x.cpnts))...), S)

include("./tensor_symmetry.jl")

(tensor_pow(arg::AbstractTensor, ::Val{1})        ) = arg
(tensor_pow(arg::AbstractTensor, ::Val{P}) where P) = arg * tensor_pow(arg, Val(P-1))

Base.:^(arg::AbstractTensor, pow::Int) = pow > 0 ? tensor_pow(arg, Val(pow)) : error("pow <= 0.")

struct TensorAdjoint{T <: AbstractTensor} <: AbstractTensor
	t::T
end

adjoint(t::AbstractTensor) = TensorAdjoint(t)

my_ndims(f::Tensor)      = filter_ndims([my_ndims(c) for c in f.cpnts])
my_ndims(f::TensorExpr)  = filter_ndims([my_ndims(c) for c in f.contents])
my_ndims(f::TensorProd)  = filter_ndims([my_ndims(c) for c in (f.t1, f.t2)])

gridsize(f::Tensor)      = filter_gridsize([gridsize(c) for c in f.cpnts])
gridsize(f::TensorExpr)  = filter_gridsize([gridsize(c) for c in f.contents])
gridsize(f::TensorProd)  = filter_gridsize([gridsize(c) for c in (f.t1, f.t2)])

include("./tensor_utils.jl")

function Tensor(cpnts::T, ::Type{S}) where {S<:TensorSymmetry, T<:NamedTuple}
	order, msg = parse_tensor(T, S)
	order >= 0 || error("When trying to parse Named Tuple as Tensor: ", msg)
	return Tensor{S,T}(cpnts)
end

function Tensor(cpnts::T) where {T<:NamedTuple}
	order, msg = parse_tensor(T)
	order >  0 || error("When trying to parse Named Tuple as Tensor: ", msg)
	return Tensor{Unsymmetric{order},T}(cpnts)
end

(Tensor(::Type{S}) where S <: TensorSymmetry) = Tensor(NamedTuple(), S)

(Tensor(dims,            stags::T        ) where {T<:NamedTuple    }) = Tensor(dims, Tensor(stags))
(Tensor(dims,            stags::Tensor{S}) where {S<:TensorSymmetry}) = Tensor(dims, S, stags.cpnts)
(Tensor(dims, ::Type{S}, stags::Tensor{S}) where {S<:TensorSymmetry}) = Tensor(dims, S, stags.cpnts)

function Tensor(dims::NTuple, ::Type{S}, stags::NamedTuple) where {S<:TensorSymmetry}
	data  = zeros(+(map(length, stags)...), dims...)
	gen   = (i, c, s) -> scalar_tensor_component(S, data, i, c, s)
	cmps  = keys(stags)
	inds  = accumulate(+, (1, map(length, values(stags))[1:end-1]...))
	vals  = map(gen, inds, cmps, values(stags))
	return Tensor(NamedTuple{cmps}(vals), S)
end

# explicitly user-facing tensor indexing
function Base.getproperty(obj::T, c::Symbol) where T <: AbstractTensor
	hasfield(T, c) && return getfield(obj, c)
	return get_component(obj, Val(c))
end

@generated function get_component(obj::Tensor{S}, c::Val{C}) where {S, C}
	cc = Val(decode_component(S, C))
	return :(symmetry_expr(S, obj, $cc))
end

function has_component(::Type{Tensor{S, T}}, c::Val{C}) where {S, C, T}
	return symmetry_expr(S, 1, Val(decode_component(S, C))) != :(0)
end
	
# tensor indexing without symmetry checking.
@generated function get_component_by_index(obj::Tensor{S, NamedTuple{N, T}}, ::Val{I}) where {S, N, T, I}
	(c = encode_component(S, I)) in N || return 0
	return :(getfield(obj.cpnts, $(Meta.quot(c))))
end

get_component_by_index(obj, ::Val) = obj
	
@generated function get_component(obj::NamedTuple{N,T}, c::Val{C}) where {N, T, C}
	C in N || error(
		"Field $C not found in NamedTuple{$N}. "*
		"Tensors offer more flexibility."
	)
	return :(getfield(obj, C))
end

function has_component(::Type{NamedTuple{N,T}}, c::Val{C}) where {N, T, C}
	return C in N
end

(get_component(p::T,      c::Val) where {T <: AbstractScalar}) = p
(has_component(::Type{T}, c::Val) where {T <: AbstractScalar}) = true

@generated function get_component(p::TensorExpr{T}, c::Val{C}) where {T <: Tuple, C}
	heads = expr_heads(fieldtypes(T)...)
	first = length(heads)+1
	last  = length(fieldtypes(T))
	args  = [:(get_component(p.contents[$i], c)) for i in first:last]
	return Expr(heads..., args...)
end

function has_component(::Type{TensorExpr{T}}, c::Val) where {T <: Tuple{Val{:call}, Val{:+}, Vararg}}
	return any([has_component(t, c) for t in fieldtypes(T)[3:end]])
end

function has_component(::Type{TensorExpr{T}}, c::Val) where {T <: Tuple{Val{:call}, Val{:-}, Vararg}}
	return any([has_component(t, c) for t in fieldtypes(T)[3:end]])
end

function has_component(::Type{TensorExpr{T}}, c::Val) where {T <: Tuple{Val{:call}, Val{:*}, Vararg}}
	return all([has_component(t, c) for t in fieldtypes(T)[3:end]])
end

function has_component(::Type{TensorExpr{T}}, c::Val) where {T <: Tuple{Val{:call}, Val{:/}, Vararg}}
	return all([has_component(t, c) for t in fieldtypes(T)[3:end]])
end

function has_component(::Type{TensorExpr{Tuple{Val{:call}, V, T}}}, c::Val) where {V <: Val, T <: AbstractTensor}
	return has_component(T, c)
end

@generated function get_component(p::TensorAdjoint{T}, c::Val{C}) where {T, C}
	cc = Val <| encode_component <| reverse <| decode_component(C)
	return :(adjoint <| get_component(p.t, $cc))
end

function has_component(::Type{TensorAdjoint{T}}, c::Val{C}) where {T, C}
	cc = Val <| encode_component <| reverse <| decode_component(C)
	return has_component(T, cc)
end

@generated function get_component(p::TensorProd{T1,T2}, c::Val{C}) where {T1, T2, C}
	oa = tensor_order(T1)
	ob = tensor_order(T2)
	oc = tensor_order(TensorProd{T1,T2})
	
	cc = div(oa + ob - oc, 2)
	
	indx = decode_component(C)
	
	ia = indx[1:oa-cc]
	ib = indx[oa-cc+1:end]
	
	Ic = combinations([1, 2, 3], cc)
	
	result = :(0)
	
	for ic in Ic
		ca = Val <| encode_component <| (ia..., ic...)
		cb = Val <| encode_component <| (ic..., ib...)
		
		if has_component(T1, ca) && has_component(T2, cb)
			aa = :(interpolate <| get_component(p.t1, $ca))
			bb = :(interpolate <| get_component(p.t2, $cb))
			result = Expr(:call, :+, result, Expr(:call, :*, aa, bb))
		end
	end
	
	return result
end

function has_component(::Type{TensorProd{T1,T2}}, c::Val{C}) where {T1, T2, C}
	oa = tensor_order(T1)
	ob = tensor_order(T2)
	oc = tensor_order(TensorProd{T1,T2})
	
	cc = div(oa + ob - oc, 2)
	
	indx = decode_component(C)
	
	ia = indx[1:oa-cc]
	ib = indx[oa-cc+1:end]
	
	Ic = combinations([1, 2, 3], cc)
	
	for ic in Ic
		ca = Val <| encode_component <| (ia..., ic...)
		cb = Val <| encode_component <| (ic..., ib...)
		
		has_component(T1, ca) && has_component(T2, cb) && return true
	end
	
	return false
end

@generated function get_component(p::T, c::Val{C}) where {T <: Tuple, C}
	args  = [:(get_component(p[$i], c)) for i in 1:length(fieldtypes(T))]
	return Expr(:tuple, args...)
end

function has_component(::Type{T}, c::Val) where {T <: Tuple}
	return any([has_component(t, c) for t in fieldtypes(T)])
end


@generated function reduce_at!(result::Ref{T}, op, field::Tensor{S, NamedTuple{N,Tt}}, inds, bounds) where {T, S, N, Tt}
	expr_rules = K -> :(reduce_at!(result, op, getfield(field.cpnts, $K), inds, bounds))
	return Expr(:block, map(expr_rules, Meta.quot.(N))...)
end

@generated function reduce_at!(result::Ref{T}, op, f1::Tensor{S, NamedTuple{N,Tt1}}, f2::Tensor{S, NamedTuple{N,Tt2}}, inds, bounds) where {T, S, N, Tt1, Tt2}
	expr_rules = K -> :(reduce_at!(result, op, getfield(f1.cpnts, $K), getfield(f2.cpnts, $K), inds, bounds))
	return Expr(:block, map(expr_rules, Meta.quot.(N))...)
end

@generated function assign_at!(lhs::Tensor{S, NamedTuple{N, T}}, rhs::RHS, inds, bounds) where {S, RHS, N, T}
	expr_rules = (k, K) -> :(assign_at!(getfield(lhs.cpnts, $K), get_component(rhs, $(Val(k))), inds, bounds))
	return Expr(:block, map(expr_rules, N, Meta.quot.(N))...)
end

@generated function assign_at!(lhs::Tensor{S, NamedTuple{N, T}}, rhs::Tuple{R, BC}, inds, bounds) where {S, N, T, R, BC <: Union{Tensor{S}, NamedTuple}}
	expr_rules = (k, K) -> :(assign_at!(getfield(lhs.cpnts, $K), (get_component(rhs[1], $(Val(k))), get_component(rhs[2], $(Val(k)))), inds, bounds))
	return Expr(:block, map(expr_rules, N, Meta.quot.(N))...)
end

(prepare_assignment(lhs::Tensor{S, NamedTuple{N, T}}, rhs) where {S, N, T}) = 
	Tensor((; zip(N, prepare_assignment(getfield(lhs.cpnts, k), get_component(rhs, Val(k))) for k in N)...), S)

(prepare_assignment(lhs::Tensor{S}, rhs::Tuple{R, BC}) where {S, R, BC}) = 
	map(r -> prepare_assignment(lhs, r), rhs)


	###########################
	##                       ##
	##    INVARIANTS ETC.    ##
	##                       ##
	###########################

L2(arg) = sqrt(
	#=interpolate(=#arg.x#=)=#^2 +
	#=interpolate(=#arg.y#=)=#^2 +
	#=interpolate(=#arg.z#=)=#^2
)

# function dot(arg1::AbstractTensor, arg2::AbstractTensor)
# 	keys  = intersect(tensor_components(arg1), tensor_components(arg2))
# 	gen  = k -> dot(
# 		#=interpolate <| =#get_component(arg1, k),
# 		#=interpolate <| =#get_component(arg2, k)
# 	)
# 	return +([gen for k in keys]...)
# end

# The following are found by the Cayley-Hamilton Theorem:
I1(arg) = (1/1) * (J1(arg)^1)
I2(arg) = (1/2) * (J1(arg)^2 - 1*J2(arg))
I3(arg) = (1/6) * (J1(arg)^3 - 3*J2(arg)*J1(arg) + 2*J3(arg))

J1(arg) = interpolate(tr(arg  ))
J2(arg) =             tr(arg^2)
J3(arg) =             tr(arg^3)

tr(arg) = arg.xx + arg.yy + arg.zz

@generated function dev(a::T) where T <: AbstractTensor
	o =  tensor_order(T)
	o == 2 && return :(a - tr(a)*Tensor(Identity)/3)
	return :(error("trying to compute deviator of rank $o tensor."))
end
	


	###########################
	##                       ##
	##    DIFF OPERATORS     ##
	##                       ##
	###########################

function grad(t::S) where S <: AbstractScalar
	return Tensor((
		x = D(t, :x),
		y = D(t, :y),
		z = D(t, :z)
	), Unsymmetric{1})
end

@generated function grad(t::T) where T <: AbstractTensor
	rank = tensor_order(T)
	rank == 1 || return :(error("gradient of rank $rank tensor not supported."))
	return :(Tensor((
		xx = D(t.x, :x),
		xy = D(t.x, :y),
		xz = D(t.x, :z),
		yx = D(t.y, :x),
		yy = D(t.y, :y),
		yz = D(t.y, :z),
		zx = D(t.z, :x),
		zy = D(t.z, :y),
		zz = D(t.z, :z)
	), Unsymmetric{2}))
end

function grad2(t::S) where S <: AbstractScalar
	return Tensor((
		xx = D(t, :x)*D(t, :x),
		xy = D(t, :y)*D(t, :x),
		yy = D(t, :y)*D(t, :y),
		xz = D(t, :z)*D(t, :x),
		yz = D(t, :z)*D(t, :y),
		zz = D(t, :z)*D(t, :z)
	), Symmetric)
end

@generated function grad2(t::T) where T <: AbstractTensor
	rank = tensor_order(T)
	rank == 1 || return :(error("gradient of rank $rank tensor not supported."))
	return :(Tensor((
		xx = D(t.x, :x)*D(t.x, :x),
		xy = D(t.x, :y)*D(t.y, :x),
		yy = D(t.y, :y)*D(t.y, :y),
		xz = D(t.x, :z)*D(t.z, :x),
		yz = D(t.y, :z)*D(t.z, :y),
		zz = D(t.z, :z)*D(t.z, :z)
	), Symmetric))
end

@generated function symgrad(t::T) where T <: AbstractTensor
	rank = tensor_order(T)
	rank == 1 || return :(error("gradient of rank $rank tensor not supported."))
	return :(Tensor((
		xx = (D(t.x, :x)),
		xy = (D(t.x, :y) + D(t.y, :x))/2,
		yy = (D(t.y, :y)),
		xz = (D(t.x, :z) + D(t.z, :x))/2,
		yz = (D(t.y, :z) + D(t.z, :y))/2,
		zz = (D(t.z, :z))
	), Symmetric))
end

# this is hijacking, but necessary at the moment to reuse the curl and symgrad operators for computing staggers
(Base.:*(a::NTuple{M, NTuple{N}}, b::Number) where {N, M}) = a
(Base.:/(a::NTuple{M, NTuple{N}}, b::Number) where {N, M}) = a
(Base.:-(a::NTuple{M, NTuple{N}}, b::NTuple{M, NTuple{N}}) where {N, M}) = a + b
function Base.:+(a::NTuple{M, NTuple{N}}...) where {N, M}
	sorted_stags = map(s -> Tuple <| sort <| [s...], a)
	all(f -> f == sorted_stags[1], sorted_stags) && return sorted_stags[1]
	error("Stag incompatibility: $a")
end

@generated function divergence(t::T) where T <: AbstractTensor
	rank = tensor_order(T)
	rank == 1 && return :(D(t.x, :x) + D(t.y, :y) + D(t.z, :z))
	rank == 2 && return :(Tensor((
		x = D(t.xx, :x) + D(t.xy, :y) + D(t.xz, :z),
		y = D(t.yx, :x) + D(t.yy, :y) + D(t.yz, :z),
		z = D(t.zx, :x) + D(t.zy, :y) + D(t.zz, :z),
	), Unsymmetric{1}))
end

@generated function curl(t::T) where T <: AbstractTensor
	rank = tensor_order(T)
	rank == 1 || return :(error("curl of rank $rank tensor not supported."))
	return :(Tensor((
		x = D(t.z, :y) - D(t.y, :z),
		y = D(t.x, :z) - D(t.z, :x),
		z = D(t.y, :x) - D(t.x, :y),
	), Unsymmetric{1}))
end


#=
curl(f) = (
		x = f_z,y - f_y,z
		y = f_x,z - f_z,x
		z = f_y,x - f_x,y
	)

curl@curl(f) = (
		x = f_y,xy + f_z,xz - f_x,yy - f_x,zz
		y = f_z,yz + f_x,yx - f_y,zz - f_y,xx
		z = f_x,zx + f_y,zy - f_z,xx - f_z,yy
	)

curl@curl(f) = (
		x = f_y,xy - f_x,yy
		y = f_x,yx - f_y,xx
	)

=#

# module StaggeredKernels