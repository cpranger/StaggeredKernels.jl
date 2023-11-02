# module StaggeredKernels

export If, A, D, BD, FD, interpolate, fieldgen, AbstractField, Field, BC, FieldRed, zero_bc, diag#, assign_at, reduce_at

abstract type AbstractField end

const Scalar = Union{Number,AbstractField}

struct Field{Indx, T <: AbstractArray} <: AbstractField
	data::T
end

(Field(dims::NTuple{N,I}, stags::NTuple{M, NTuple{N,I}}) where {M, N, I  <: Int}) =
    Field{stags}(zeros(M, dims...))
	
(Field{SS}(data::T) where {SS, T <: AbstractArray}) =
    Field{SS, T}(data)

# TODO: TEST DEEP VS SHALLOW COPY WHEN CONSTRUCTING
struct FieldExpr{T <: Tuple} <: AbstractField
	contents::T
end

# (FieldExpr(args::T) where T <: Tuple) = FieldExpr{T}(args)
(FieldExpr(args...)) = FieldExpr(tuple(args...))

FieldOp(op::Symbol, args...) = FieldExpr(Val(:call), Val(op), args...)
If(c::FieldExpr, t, f = 0) = FieldExpr(Val(:if), c, t, f)

struct FieldShft{S, T <: AbstractField} <: AbstractField
	shiftee::T
end

(S(arg::T, shft::NTuple) where T) = length(shft) == my_ndims(arg) ? FieldShft{shft,T}(arg) : error("Dimension mismatch.")

kronecker(i, j) = Int(i == j)
kronecker(i) = n -> Tuple([kronecker(i, j) for j in 1:n])
# stagdir(stag, dir, rad) = Tuple([stag[k] + rad*kronecker(k,dir) for k in 1:length(stag)])

 D(x::Number, ::Symbol) = 0
BD(x::Number, ::Symbol) = 0
FD(x::Number, ::Symbol) = 0
 A(x::Number, ::Symbol) = x

 D(x, d::Symbol) =  D(x, decode_component(d)[1])
BD(x, d::Symbol) = BD(x, decode_component(d)[1])
FD(x, d::Symbol) = FD(x, decode_component(d)[1])
 A(x, d::Symbol) =  A(x, decode_component(d)[1])

 D(x, d::Int   ) = d <= my_ndims(x) ?  D(x, kronecker(d)(my_ndims(x))) : 0
BD(x, d::Int   ) = d <= my_ndims(x) ? BD(x, kronecker(d)(my_ndims(x))) : 0
FD(x, d::Int   ) = d <= my_ndims(x) ? FD(x, kronecker(d)(my_ndims(x))) : 0
 A(x, d::Int   ) = d <= my_ndims(x) ?  A(x, kronecker(d)(my_ndims(x))) : 0

 D(x::AbstractField, d::NTuple) = (S(x, .+d) - S(x, .-d))
BD(x::AbstractField, d::NTuple) =  S(D(x, d), .-d)
FD(x::AbstractField, d::NTuple) =  S(D(x, d), .+d)
 A(x::AbstractField, d::NTuple) = (S(x,    .+d) + S(x,    .-d))/2

( D(x::NTuple{M,NTuple{N,I}}, d::NTuple) where {N, M, I <: Int}) = Tuple <| unique <| map(s -> mod.(s .+ d, 2), x)
( A(x::NTuple{M,NTuple{N,I}}, d::NTuple) where {N, M, I <: Int}) = Tuple <| unique <| map(s -> mod.(s .+ d, 2), x)


struct FieldIntp{T} <: AbstractField
	interpolant::T
end

interpolate(arg::Number) = arg

function interpolate(arg::T) where T <: AbstractField
	s = stags(T)
	length(s) == 1 || error("Currently only one stag supported per field.")
	return FieldIntp{T}(arg)
end

struct FieldGen{F <: Function} <: AbstractField
	func::F
end

fieldgen(f::Function) = FieldGen(f)

struct FieldDiag{F, SS, O} <: AbstractField
	f::F
	x::Field{SS}
end

(diag(x::Field{SS}, f::F, offset = 0 .* SS[1]) where {F, SS}) = FieldDiag{F, SS, offset}(f, x)

function diag(x::Field{SS}, f::Tuple{F, BCs}, offset = 0 .* SS[1]) where {F, BCs, SS}
	bcs = parse_bcs(f[2])
	FieldDiag{Tuple{F, typeof(bcs)}, SS, offset}((f[1], bcs), x)
end

struct BC{D, T <: Scalar}
	expr::T
end

function parse_bc_dir(signaxis::String)
	sign, axis = split(signaxis, "")
	
	sign_dict  = Dict(["-" => -1, "+" => +1])
	axis_dict  = Dict(["x" =>  1, "y" =>  2, "z" =>  3])
	
	sign in keys(sign_dict) || error( "first character should be in $(keys(sign_dict)).")
	axis in keys(axis_dict) || error("second character should be in $(keys(axis_dict)).")
	
	return sign_dict[sign] * axis_dict[axis]
end

(BC{D}(val::T) where {D, T <: Scalar}) = BC{D,T}(val)
(BC(d::String, val::T) where T <: Scalar) = BC{parse_bc_dir(d), T}(val)

parse_bcs(arg::Pair)        = BC(arg[1], arg[2])
parse_bcs(args::Tuple)      = map(parse_bcs, args)
parse_bcs(args::NamedTuple) = (; zip(keys(args), map(v -> parse_bcs(v), values(args)))...)

combinations(v::Vector, n::Int) = combinations(fill(v, n)...)

function combinations(vecs::Vector...)
	l = length(vecs)
	shapes = [[i == j ? (:) : 1 for j in 1:l] for i in 1:l]
	result = tuple.(map((vec, shape) -> reshape(vec, shape...), vecs, shapes)...)
	return reshape(result, *(length.(vecs)...))
end

function filter_ndims(a::Vector)
	all(i -> i == 0 || i == max(a...), a) && return max(a...)
	error("Incompatible field dimensionalities ($a).")
end

my_ndims(f::Field)      = ndims(f.data) - 1
my_ndims(f::FieldExpr)  = filter_ndims([my_ndims(c) for c in f.contents])
my_ndims(f::FieldShft)  = my_ndims(f.shiftee)
my_ndims(f::FieldIntp)  = my_ndims(f.interpolant)
my_ndims(f::FieldGen)   = first(methods(f.func)).nargs
(my_ndims(f::NTuple{M,NTuple{N}}) where {M, N}) = N
my_ndims(f::Any)        = 0

(stags(::Type{Field{St, T}})   where {St, T}) = St
(stags(::Type{FieldExpr{Tt}})  where {Tt        }) = intersect([stags(t) for t in fieldtypes(Tt)]...)
(stags(::Type{FieldShft{S,T}}) where {S,  T     }) = [mod.(s .+ S, 2) for s in stags(T)]
(stags(::Type{FieldIntp{A}})   where {A         }) = combinations([0, 1], length(stags(A)[1]))
(stags(::Type)) = union(map(n -> combinations([0, 1], n), 1:3)...)

stencil_offset(s::(NTuple{N,Int} where N)) = div.(s, 2, RoundUp)

isect_bounds(bounds1, bounds2) = (
	Tuple(map(max, bounds1[1], bounds2[1])),
	Tuple(map(min, bounds1[2], bounds2[2]))
)

function stagindex(ss, s)
	s in ss || error("Stag $s not in collection $ss.")
	i = findfirst(a -> a == s, ss)
	i == findlast(a -> a == s, ss) ||
	           error("Stag $s encountered more than once in $ss.")
	return i
end

getindex(x::Number, ::Val, inds, bounds) = x

Base.getindex( x::Field,      inds...) =  x.data[inds...]
Base.setindex!(x::Field, val, inds...) = (x.data[inds...] = val)

@generated function getindex(x::Field{SS}, ::Val{S}, inds, bounds) where {SS, S}
	f = stagindex(SS, mod.(S, 2))
	offset = stencil_offset(S)
	return :(x[$f, min.(max.(bounds[1], inds .+ $offset), bounds[2] .+ $(mod.(S, 2) .- 1))...]) # This imposes automatic Neumann conditions whenever needed.
end

@generated function setindex!(x::Field{SS}, val, ::Val{S}, inds, bounds) where {SS, S}
	f = stagindex(SS, mod.(S, 2))
	offset = stencil_offset(S)
	return :(x[$f, min.(max.(bounds[1], inds .+ $offset), bounds[2] .+ $(mod.(S, 2) .- 1))...] = val) # This imposes automatic Neumann conditions whenever needed.
end

(expr_heads(args...)) = []
(expr_heads(::Type{Val{S}}, args...) where S) = [S, expr_heads(args...)...]

@generated function getindex(x::FieldExpr{T}, s::Val{S}, inds, bounds) where {T <: Tuple, S}
	heads = expr_heads(fieldtypes(T)...)
	first = length(heads)+1
	last  = length(fieldtypes(T))
	args  = [:(getindex(x.contents[$i], s, inds, bounds)) for i in first:last]
	return Expr(heads..., args...)
end

(getindex(x::FieldShft{S1,T}, ::Val{S2}, inds, bounds) where {T, S1, S2}) = 
	getindex(x.shiftee, Val(S1 .+ S2), inds, bounds)

(getindex(x::FieldGen, s::Val{S}, inds, bounds) where S) = x.func((inds .+ S./2 .- 1)...)

@generated function getindex(x::FieldIntp{A}, ::Val{S}, inds, bounds) where {A, S}
	sten = combinations(map((f, t) -> mod(f,2) == mod(t,2) ? [t] : [t-1,t+1], stags(A)[1], S)...)
	size = length(sten)
	args = [:(getindex(x.interpolant, $(Val(s)), inds, bounds)) for s in sten]
	return Expr(:call, :/, Expr(:call, :+, args...), size)
end

@generated function getindex(diag::FieldDiag{F, SS, O}, ::Val{S}, inds, bounds) where {F, SS, O, S}
	f = stagindex(SS, mod.(S, 2))
	s = (Val(.-mod.(S, 2)))
	return :(
		setindex!(diag.x, 1, $s, inds .+ $O, bounds);
		r = getindex(diag.f, $s, inds,       bounds);
		setindex!(diag.x, 0, $s, inds .+ $O, bounds);
		r
	)
end

(getindex(f::Tuple{F, BCs}, s::Val{S}, inds, bounds) where {F, BCs <: Tuple, S}) = 
	getindex((f[2]..., f[1]), s, inds, bounds)

getindex(f::Tuple{Scalar}, s, inds, bounds) = getindex(f[1], s, inds, bounds)

@generated function getindex(args::Tuple{BC{D}, Union{Scalar, BC}, Vararg{Union{Scalar, BC}}}, s::Val{S}, inds, bounds) where {D, S}
	d = abs(D)
	f = mod(-sign(D), 3) # (-n, +m) -> (1, 2) ∀ n, m ∈ N+
	o = Int(f == 2) * (mod(S[d], 2) - 1)
	return :(
		inds[$d] == (bounds[$f][$d] + $o) ?
			getindex(args[1].expr, s, inds, bounds) :
			getindex(args[2:end],  s, inds, bounds)
	)
end

@generated function reduce_at!(result::Ref{T}, op, f::Field{S}, inds, bounds) where {T, S}
	return Expr(
		:block,
		[Expr(
			:if,
			:(all(bounds[1] .<= inds .<= bounds[2] .+ $(mod.(s, 2) .- 1))),
			:(result[] = op(
				result[],
				getindex(f, $(Val(.-s)), inds, bounds)
			))
		) for s in S]...
	)
end

@generated function reduce_at!(result::Ref{T}, op, f1::Field{S}, f2::Field{S}, inds, bounds) where {T, S}
	return Expr(
		:block,
		[Expr(
			:if,
			:(all(bounds[1] .<= inds .<= bounds[2] .+ $(mod.(s, 2) .- 1))),
			:(result[] = op(
				result[],
				getindex(f1, $(Val(.-s)), inds, bounds),
				getindex(f2, $(Val(.-s)), inds, bounds)
			))
		) for s in S]...
	)
end

@generated function assign_at!(lhs::Field{S}, rhs, inds, bounds) where S
	return Expr(
		:block,
		[:(assign_at!(
			lhs,
			rhs,
			$(Val(.-s)),
			inds,
			bounds
		)) for s in S]...
	)
end

#    x   |   x       x       x       x       x       x       x   |   x    
#        1       2       3       4       5       6       7       8       9
#        x       x       x       x       x       x       x       x       -
#        |                                                       |        
#    x   |   x       x       x       x       x       x       x   |   x    
#    1   |   2       3       4       5       6       7       8   |   9    
#        x       x       x       x       x       x       x       x       -
#        |                                                       |         

@generated function assign_at!(lhs::Field{Stags}, rhs, s::Val{Stag}, inds, bounds) where {Stags, Stag}
	f = stagindex(Stags, mod.(Stag, 2))
	return :(all(inds .<= (bounds[2] .+ $(mod.(Stag, 2) .- 1))) ? (lhs[$f, inds...] = getindex(rhs, s, inds, bounds)) : ())
end

Base.:(+)(a::AbstractField) = a
Base.:(-)(a::AbstractField) = FieldOp(:-, a)

Base.sqrt(a::AbstractField) = FieldOp(:sqrt, a)
Base.imag(a::AbstractField) = FieldOp(:imag, a)
Base.real(a::AbstractField) = FieldOp(:real, a)

Base.abs(a::AbstractField) = FieldOp(:abs, a)
Base.exp(a::AbstractField) = FieldOp(:exp, a)
Base.sin(a::AbstractField) = FieldOp(:sin, a)
Base.cos(a::AbstractField) = FieldOp(:cos, a)

Base.:(+)(a::AbstractField, b::AbstractField) = FieldOp(:+, a, b)
Base.:(+)(a::Scalar, b::AbstractField)  = FieldOp(:+, a, b)
Base.:(+)(a::AbstractField, b::Scalar)  = FieldOp(:+, a, b)

Base.:(-)(a::AbstractField, b::AbstractField) = FieldOp(:-, a, b)
Base.:(-)(a::Scalar, b::AbstractField)  = FieldOp(:-, a, b)
Base.:(-)(a::AbstractField, b::Scalar)  = FieldOp(:-, a, b)

Base.:(*)(a::AbstractField, b::AbstractField) = FieldOp(:*, a, b)
Base.:(*)(a::Scalar, b::AbstractField)  = FieldOp(:*, a, b)
Base.:(*)(a::AbstractField, b::Scalar)  = FieldOp(:*, a, b)

Base.:(/)(a::AbstractField, b::AbstractField) = FieldOp(:/, a, b)
Base.:(/)(a::Scalar, b::AbstractField)  = FieldOp(:/, a, b)
Base.:(/)(a::AbstractField, b::Scalar)  = FieldOp(:/, a, b)

Base.:(^)(a::AbstractField, b::AbstractField) = FieldOp(:^, a, b)
Base.:(^)(a::Scalar, b::AbstractField)  = FieldOp(:^, a, b)
Base.:(^)(a::AbstractField, b::Scalar)  = FieldOp(:^, a, b)

Base.:(<)(a::AbstractField, b::AbstractField) = FieldOp(:(<), [a, b])
Base.:(<)(a::Scalar, b::AbstractField)  = FieldOp(:(<),  a, b)
Base.:(<)(a::AbstractField, b::Scalar)  = FieldOp(:(<),  a, b)

Base.:(<=)(a::AbstractField, b::AbstractField) = FieldOp(:(<=), a, b)
Base.:(<=)(a::Scalar, b::AbstractField) = FieldOp(:(<=), a, b)
Base.:(<=)(a::AbstractField, b::Scalar) = FieldOp(:(<=), a, b)

Base.:(>)(a::AbstractField, b::AbstractField) = FieldOp(:(>), a, b)
Base.:(>)(a::Scalar, b::AbstractField)  = FieldOp(:(>),  a, b)
Base.:(>)(a::AbstractField, b::Scalar)  = FieldOp(:(>),  a, b)

Base.:(>=)(a::AbstractField, b::AbstractField) = FieldOp(:(>=), a, b)
Base.:(>=)(a::Scalar, b::AbstractField) = FieldOp(:(>=), a, b)
Base.:(>=)(a::AbstractField, b::Scalar) = FieldOp(:(>=), a, b)

Base.:(==)(a::AbstractField, b::AbstractField) = FieldOp(:(==), a, b)
Base.:(==)(a::Scalar, b::AbstractField) = FieldOp(:(==), a, b)
Base.:(==)(a::AbstractField, b::Scalar) = FieldOp(:(==), a, b)

# dot(a::AbstractField, b::AbstractField) = interpolate(a) * interpolate(b)

# module StaggeredKernels