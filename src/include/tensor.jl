# module StaggeredKernels

export Tensor, Vector, L2, J1, J2, J3, I1, I2, I3, tr, dev, divergence, grad, symgrad, curl

abstract type AbstractTensor end

abstract type TensorSymmetry{O} end

struct Tensor{S<:TensorSymmetry, T <: NamedTuple} <: AbstractTensor
	cpnts::T
end

struct TensorProd{T1 <: AbstractTensor, T2 <: AbstractTensor} <: AbstractTensor
	t1::T1
	t2::T2
end

struct TensorScale{T <: AbstractTensor, S <: Scalar} <: AbstractTensor
	t::T
	s::S
end

include("./tensor_symmetry.jl")

Base.:*(a::AbstractTensor, b::AbstractTensor) =  TensorProd(a,   b)
Base.:*(a::AbstractTensor, b::Scalar        ) = TensorScale(a,   b)
Base.:*(a::Scalar        , b::AbstractTensor) = TensorScale(b,   a)
Base.:/(a::AbstractTensor, b::Scalar        ) = TensorScale(a, 1/b)
Base.:-(a::AbstractTensor)                    = TensorScale(a,  -1)

(nt_prod(a::Scalar, b::NamedTuple{N}) where {N}) = NamedTuple{N}(map(v -> a * v, values(b)))
(nt_prod(a::NamedTuple{N}, b::Scalar) where {N}) = NamedTuple{N}(map(v -> b * v, values(a)))

# (Base.:*(a::Scalar, b::Tensor{S}) where S) = Tensor(nt_prod(  a, b.cpnts), S)
# (Base.:*(a::Tensor{S}, b::Scalar) where S) = Tensor(nt_prod(  b, a.cpnts), S)
# (Base.:/(a::Tensor{S}, b::Scalar) where S) = Tensor(nt_prod(1/b, a.cpnts), S)

Base.:/(a::AbstractTensor, b::Tensor{Unsymmetric{1}}) = Vector((x = a.x/b.x, y = a.y/b.y, z = a.z/b.z,))

(tensor_pow(arg::AbstractTensor, ::Val{1})        ) = arg
(tensor_pow(arg::AbstractTensor, ::Val{P}) where P) = arg * tensor_pow(arg, Val(P-1))

Base.:^(arg::AbstractTensor, pow::Int) = pow > 0 ? tensor_pow(arg, Val(pow)) : error("pow <= 0.")


struct TensorSum{T1 <: AbstractTensor, T2 <: AbstractTensor} <: AbstractTensor
	t1::T1
	t2::T2
end

Base.:+(a::AbstractTensor, b::AbstractTensor) = TensorSum(a, b)
Base.:-(a::AbstractTensor, b::AbstractTensor) = a + (-b)

# struct TensorNeg{T <: AbstractTensor} <: AbstractTensor
# 	t::T
# end


struct TensorAdjoint{T <: AbstractTensor} <: AbstractTensor
	t::T
end

adjoint(t::AbstractTensor) = TensorAdjoint(t)


include("./tensor_utils.jl")

function Tensor(cpnts::T, ::Type{S}) where {S<:TensorSymmetry, T<:NamedTuple}
	order, msg = parse_tensor(T, S)
	order >= 0 || error("When trying to parse Named Tuple as Tensor: ", msg)
	return Tensor{S,T}(cpnts)
end

(Vector(cpnts::T) where {T<:NamedTuple}) = Tensor{Unsymmetric{1},T}(cpnts)

function Tensor(cpnts::T) where {T<:NamedTuple}
	order, msg = parse_tensor(T)
	order >  0 || error("When trying to parse Named Tuple as Tensor: ", msg)
	return Tensor{S,T}(cpnts, Unsymmetric{order})
end

(Tensor(::Type{S}) where S <: TensorSymmetry) = Tensor(NamedTuple(), S)

(Tensor(dims, ::Type{S}, stags::Tensor, bcs = NamedTuple()) where {S<:TensorSymmetry}) = 
	Tensor(dims, S, stags.cpnts, bcs)

function Tensor(dims::NTuple, ::Type{S}, stags::NamedTuple, bcs = NamedTuple()) where {S<:TensorSymmetry}
	data  = @zeros(+(map(length, stags)...), dims...)
	gen   = (i, c, s) -> scalar_tensor_component(S, data, bcs, i, c, s)
	cmps  = keys(stags)
	inds  = accumulate(+, (1, map(length, values(stags))[1:end-1]...))
	vals  = map(gen, inds, cmps, values(stags))
	return Tensor(NamedTuple{cmps}(vals), S)
end

Vector(dims, stags, bcs = NamedTuple()) = Tensor(dims, Unsymmetric{1}, stags, bcs)

# explicitly user-facing tensor indexing
function Base.getproperty(obj::T, c::Symbol) where T <: AbstractTensor
	hasfield(T, c) && return getfield(obj, c)
	return get_component(obj, Val(c))
end

@generated function get_component(obj::Tensor{S}, c::Val{C}) where {S, C}
	cc = Val(decode_component(S, C))
	return :(symmetry_expr(S, obj, $cc))
end
	
# tensor indexing without symmetry checking.
@generated function get_component_by_index(obj::Tensor{S, NamedTuple{N, T}}, ::Val{I}) where {S, N, T, I}
	(c = encode_component(S, I)) in N || return 0
	return :(getfield(obj.cpnts, $(Meta.quot(c))))
end
	
@generated function get_component(obj::NamedTuple{N,T}, c::Val{C}) where {N, T, C}
	C in N || error(
		"Field $C not found in NamedTuple{$N}. "*
		"Tensors offer more flexibility."
	)
	return :(getfield(obj, C))
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
		
		aa = :(interpolate <| get_component(p.t1, $ca))
		bb = :(interpolate <| get_component(p.t2, $cb))
		result = Expr(:call, :+, result, Expr(:call, :*, aa, bb))
	end
	
	return result
end

(get_component(p::TensorScale{T, S}, c::Val{C}) where {T, S, C}) = p.s * get_component(p.t, c)

(get_component(p::TensorSum{T1, T2}, c::Val{C}) where {T1, T2, C}) = get_component(p.t1, c) + get_component(p.t2, c)

@generated function get_component(p::TensorAdjoint{T}, c::Val{C}) where {T, C}
	cc = Val <| encode_component <| reverse <| decode_component(C)
	return :(adjoint <| get_component(p.t, $cc))
end


@generated function reduce_at!(result::AbstractArray{T,0}, op, field::Tensor{S, NamedTuple{N,Tt}}, inds, bounds) where {T, S, N, Tt}
	keys  = field.parameters[1]
	expr_rules = K -> :(reduce_at!(result, op, getfield(field.cpnts, $K), inds, bounds))
	return Expr(:block, map(expr_rules, Meta.quot.(N))...)
end

@generated function reduce_at!(result::AbstractArray{T,0}, op, f1::Tensor{S, NamedTuple{N,Tt1}}, f2::Tensor{S, NamedTuple{N,Tt2}}, inds, bounds) where {T, S, N, Tt1, Tt2}
	keys  = f1.parameters[1]
	expr_rules = K -> :(reduce_at!(result, op, getfield(f1.cpnts, $K), getfield(f2.cpnts, $K), inds, bounds))
	return Expr(:block, map(expr_rules, Meta.quot.(N))...)
end

@generated function assign_at!(lhs::Tensor{S, NamedTuple{N, T}}, rhs, inds, bounds) where {S, N, T}
	expr_rules = (k, K) -> :(assign_at!(getfield(lhs.cpnts, $K), get_component(rhs, $(Val(k))), inds, bounds))
	return Expr(:block, map(expr_rules, N, Meta.quot.(N))...)
end


	###########################
	##                       ##
	##    INVARIANTS ETC.    ##
	##                       ##
	###########################

L2(arg) = sqrt(
	interpolate(arg.x)^2 +
	interpolate(arg.y)^2 +
	interpolate(arg.z)^2
)

function dot(arg1::AbstractTensor, arg2::AbstractTensor)
	keys  = intersect(tensor_components(arg1), tensor_components(arg2))
	gen  = k -> dot(
		interpolate <| get_component(arg1, k),
		interpolate <| get_component(arg2, k)
	)
	return +([gen for k in keys]...)
end

# The following are found by the Cayley-Hamilton Theorem:
I1(arg) = (1/1) * (J1(arg)^1)
I2(arg) = (1/2) * (J1(arg)^2 - 1*J2(arg))
I3(arg) = (1/6) * (J1(arg)^3 - 3*J2(arg)*J1(arg) + 2*J3(arg))

J1(arg) = interpolate(tr(arg))
J2(arg) = tr(arg^2)
J3(arg) = tr(arg^3)

tr(arg) = arg.xx + arg.yy + arg.zz

function dev(a::AbstractTensor)
	o =  tensor_order(a)
	o == 2 && return a - tr(a)*Tensor(Identity)/3
	error("trying to compute deviator of rank $o tensor.")
end
	


	###########################
	##                       ##
	##    DIFF OPERATORS     ##
	##                       ##
	###########################

function grad(t)
	return Tensor((
		x = D(t, :x),
		y = D(t, :y),
		z = D(t, :z)
	), Unsymmetric{1})
end

function grad(t::AbstractTensor)
	rank = tensor_order(t)
	rank == 1 || error("gradient of rank $rank tensor not supported.")
	return Tensor((
		xx = D(t.x, :x),
		xy = D(t.x, :y),
		xz = D(t.x, :z),
		yx = D(t.y, :x),
		yy = D(t.y, :y),
		yz = D(t.y, :z),
		zx = D(t.z, :x),
		zy = D(t.z, :y),
		zz = D(t.z, :z)
	), Unsymmetric{2})
end

function symgrad(t::AbstractTensor)
	rank = tensor_order(t)
	rank == 1 || error("gradient of rank $rank tensor not supported.")
	return Tensor((
		xx = (D(t.x, :x)),
		xy = (D(t.x, :y) + D(t.y, :x))/2,
		yy = (D(t.y, :y)),
		xz = (D(t.x, :z) + D(t.z, :x))/2,
		yz = (D(t.y, :z) + D(t.z, :y))/2,
		zz = (D(t.z, :z))
	), Symmetric)
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

function divergence(t::AbstractTensor)
	rank = tensor_order(t)
	rank == 1 && return D(t.x, :x) + D(t.y, :y) + D(t.z, :z)
	rank == 2 && return Tensor((
		x = D(t.xx, :x) + D(t.xy, :y) + D(t.xz, :z),
		y = D(t.yx, :x) + D(t.yy, :y) + D(t.yz, :z),
		z = D(t.zx, :x) + D(t.zy, :y) + D(t.zz, :z),
	), Unsymmetric{1})
end

function curl(t::AbstractTensor)
	rank = tensor_order(t)
	rank == 1 && return Tensor((
		x = D(t.z, :y) - D(t.y, :z),
		y = D(t.x, :z) - D(t.z, :x),
		z = D(t.y, :x) - D(t.x, :y),
	), Unsymmetric{1})
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