# module StaggeredKernels

export If, A, D, BD, FD, BA, FA, interpolate, fieldgen, AbstractScalarField, Field, FieldVal, BC, diag#, assign_at, reduce_at

abstract type AbstractScalarField end

const AbstractScalar = Union{Number, AbstractScalarField}

struct Field{Indx, T <: AbstractArray} <: AbstractScalarField
	data::T
end
(Field(dims::NTuple{N,I}, stags::NTuple{M, NTuple{N,I}}) where {M, N, I  <: Int}) =
    Field{stags}(zeros(M, dims...))
	
(Field{SS}(data::T) where {SS, T <: AbstractArray}) =
    Field{SS, T}(data)

struct FieldVal{R <: Ref} <: AbstractScalarField
	ref::R
end

# TODO: TEST DEEP VS SHALLOW COPY WHEN CONSTRUCTING
struct FieldExpr{T <: Tuple} <: AbstractScalarField
	contents::T
end

# (FieldExpr(args::T) where T <: Tuple) = FieldExpr{T}(args)
(FieldExpr(args...)) = FieldExpr(tuple(args...))

ScalarOp(op::Symbol, args...) = FieldExpr(Val(:call), Val(op), args...)
If(c::FieldExpr, t, f = 0) = FieldExpr(Val(:if), c, t, f)

struct FieldShft{S, T <: AbstractScalarField} <: AbstractScalarField
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
BA(x::Number, ::Symbol) = x
FA(x::Number, ::Symbol) = x

 D(x::FieldVal, ::Symbol) = 0
BD(x::FieldVal, ::Symbol) = 0
FD(x::FieldVal, ::Symbol) = 0
 A(x::FieldVal, ::Symbol) = x
BA(x::FieldVal, ::Symbol) = x
FA(x::FieldVal, ::Symbol) = x

 D(x, d::Symbol) =  D(x, decode_component(d)[1])
BD(x, d::Symbol) = BD(x, decode_component(d)[1])
FD(x, d::Symbol) = FD(x, decode_component(d)[1])
 A(x, d::Symbol) =  A(x, decode_component(d)[1])
BA(x, d::Symbol) = BA(x, decode_component(d)[1])
FA(x, d::Symbol) = FA(x, decode_component(d)[1])

 D(x, d::Int   ) = d <= my_ndims(x) ?  D(x, kronecker(d)(my_ndims(x))) : 0
BD(x, d::Int   ) = d <= my_ndims(x) ? BD(x, kronecker(d)(my_ndims(x))) : 0
FD(x, d::Int   ) = d <= my_ndims(x) ? FD(x, kronecker(d)(my_ndims(x))) : 0
 A(x, d::Int   ) = d <= my_ndims(x) ?  A(x, kronecker(d)(my_ndims(x))) : 0
BA(x, d::Int   ) = d <= my_ndims(x) ? BA(x, kronecker(d)(my_ndims(x))) : 0
FA(x, d::Int   ) = d <= my_ndims(x) ? FA(x, kronecker(d)(my_ndims(x))) : 0

 D(x::AbstractScalarField, d::NTuple) = (S(x, .+d) - S(x, .-d))
BD(x::AbstractScalarField, d::NTuple) =  S(D(x, d), .-d)
FD(x::AbstractScalarField, d::NTuple) =  S(D(x, d), .+d)
 A(x::AbstractScalarField, d::NTuple) = (S(x,    .+d) + S(x,    .-d))/2
BA(x::AbstractScalarField, d::NTuple) =  S(A(x, d), .-d)
FA(x::AbstractScalarField, d::NTuple) =  S(A(x, d), .+d)

( D(x::NTuple{M,NTuple{N,I}}, d::NTuple) where {N, M, I <: Int}) = Tuple <| unique <| map(s -> mod.(s .+ d, 2), x)
( A(x::NTuple{M,NTuple{N,I}}, d::NTuple) where {N, M, I <: Int}) = Tuple <| unique <| map(s -> mod.(s .+ d, 2), x)


struct FieldIntp{T} <: AbstractScalarField
	interpolant::T
end

interpolate(arg::Number) = arg

function interpolate(arg::T) where T <: AbstractScalarField
	s = stags(T)
	length(s) == 1 || error("Currently only one stag supported per field.")
	return FieldIntp{T}(arg)
end

struct FieldGen{F <: Function} <: AbstractScalarField
	func::F
end

fieldgen(f::Function) = FieldGen(f)

struct FieldDiag{F, SS, O} <: AbstractScalarField
	f::F
	x::Field{SS}
end

(diag(x::Field{SS}, f::F, offset = 0 .* SS[1]) where {F, SS}) = FieldDiag{F, SS, offset}(f, x)

struct BC{D, T <: AbstractScalar}
	expr::T
end

(BC{D}(val::T) where {D,  T <: AbstractScalar}) = BC{D,T}(val)
(BC(d::Int, val::T) where T <: AbstractScalar ) = BC{d,T}(val)

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
my_ndims(f::FieldVal)   = 0
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

@inline getindex(x::Number, ::Val, inds, bounds) = x

@inline Base.getindex( x::Field,      inds...) =  x.data[inds...]
@inline Base.setindex!(x::Field, val, inds...) = (x.data[inds...] = val)

@inline @generated function getindex(x::Field{SS}, ::Val{S}, inds, bounds) where {SS, S}
	f = stagindex(SS, mod.(S, 2))
	offset = stencil_offset(S)
	return :(x[$f, min.(max.(bounds[1], inds .+ $offset), bounds[2] .+ $(mod.(S, 2) .- 1))...]) # This imposes automatic Neumann conditions whenever needed.
end

@inline @generated function setindex!(x::Field{SS}, val, ::Val{S}, inds, bounds) where {SS, S}
	f = stagindex(SS, mod.(S, 2))
	offset = stencil_offset(S)
	return :(x[$f, min.(max.(bounds[1], inds .+ $offset), bounds[2] .+ $(mod.(S, 2) .- 1))...] = val) # This imposes automatic Neumann conditions whenever needed.
end

(expr_heads(args...)) = []
(expr_heads(::Type{Val{S}}, args...) where S) = [S, expr_heads(args...)...]

@inline @generated function getindex(x::FieldExpr{T}, s::Val{S}, inds, bounds) where {T <: Tuple, S}
	heads = expr_heads(fieldtypes(T)...)
	first = length(heads)+1
	last  = length(fieldtypes(T))
	args  = [:(getindex(x.contents[$i], s, inds, bounds)) for i in first:last]
	return Expr(heads..., args...)
end

@inline Base.getindex( x::FieldVal) = x.ref[]
@inline Base.setindex!(x::FieldVal, val) = Base.setindex!(x.ref, val)
@inline (getindex( x::FieldVal{R}, s::Val{S}, inds, bounds) where {R <: Ref, S}) = x.ref[]

@inline (getindex(x::FieldShft{S1,T}, ::Val{S2}, inds, bounds) where {T, S1, S2}) = 
	getindex(x.shiftee, Val(S1 .+ S2), inds, bounds)

@inline (getindex(x::FieldGen, s::Val{S}, inds, bounds) where S) = x.func((inds .+ S./2 .- 1)...)

@inline @generated function getindex(x::FieldIntp{A}, ::Val{S}, inds, bounds) where {A, S}
	sten = combinations(map((f, t) -> mod(f,2) == mod(t,2) ? [t] : [t-1,t+1], stags(A)[1], S)...)
	size = length(sten)
	args = [:(getindex(x.interpolant, $(Val(s)), inds, bounds)) for s in sten]
	return Expr(:call, :/, Expr(:call, :+, args...), size)
end

@inline @generated function getindex(diag::FieldDiag{F, SS, O}, ::Val{S}, inds, bounds) where {F, SS, O, S}
	f = stagindex(SS, mod.(S, 2))
	s = (Val(.-mod.(S, 2)))
	return :(
		setindex!(diag.x, 1, $s, inds .+ $O, bounds);
		r = getindex(diag.f, $s, inds,       bounds);
		setindex!(diag.x, 0, $s, inds .+ $O, bounds);
		r
	)
end

@inline (getindex(f::Tuple{F, BCs}, s::Val{S}, inds, bounds) where {F, BCs <: Tuple, S}) = 
	getindex_impl(s, inds, bounds, f...)

@inline getindex_impl(s::Val, inds, bounds, f::AbstractScalar) = getindex(f, s, inds, bounds)
@inline getindex_impl(s::Val, inds, bounds, f::AbstractScalar, args::Tuple) = getindex_impl(s, inds, bounds, args..., f)

@inline @generated function getindex_impl(s::Val{S}, inds, bounds, arg::BC{D}, args...) where {D, S}
	d = abs(D)
	f = mod(-sign(D), 3) # (-n, +m) -> (1, 2) ∀ n, m ∈ N+
	o = mod.(S, 2) .- 1
	return :(
		inds[$d] == (bounds[$f][$d] + $(Int(f == 2) * o[d])) ? (
				getindex(arg.expr, s, inds, bounds)
			) : getindex_impl(s, inds, bounds, args...)
	)
end

@inline @generated function reduce_at!(result::Ref{T}, op, f::Field{S}, inds, bounds) where {T, S}
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

@inline (reduce_at!(result::Ref{R}, op::OP, f1::F1, f2::F2, inds, bounds) where {R, OP, F1 <: Field, F2 <: Field}) =
	reduce_at_impl_!(result, op, f2, f1, inds, bounds)
@inline (reduce_at!(result::Ref{R}, op::OP, f1::F1, f2::F2, inds, bounds) where {R, OP, F1 <: AbstractScalarField, F2 <: Field}) =
	reduce_at_impl_!(result, op, f2, f1, inds, bounds)
@inline (reduce_at!(result::Ref{R}, op::OP, f1::F1, f2::F2, inds, bounds) where {R, OP, F1 <: Field, F2 <: AbstractScalarField}) =
	reduce_at_impl_!(result, op, f1, f2, inds, bounds)

@inline @generated function reduce_at_impl_!(result::Ref{R}, op, f1::Field{S}, f2::AbstractScalarField, inds, bounds) where {R, S}
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

@inline @generated function assign_at!(lhs::Field{S}, rhs, inds, bounds) where S
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

prepare_assignment(lhs::Field, rhs) = rhs

#    x   |   x       x       x       x       x       x       x   |   x    
#        1       2       3       4       5       6       7       8       9
#        x       x       x       x       x       x       x       x       -
#        |                                                       |        
#    x   |   x       x       x       x       x       x       x   |   x    
#    1   |   2       3       4       5       6       7       8   |   9    
#        x       x       x       x       x       x       x       x       -
#        |                                                       |         

@inline @generated function assign_at!(lhs::Field{Stags}, rhs::RHS, s::Val{Stag}, inds, bounds) where {Stags, RHS, Stag}
	f = stagindex(Stags, mod.(Stag, 2))
	return :(all(inds .<= (bounds[2] .+ $(mod.(Stag, 2) .- 1))) ? (lhs[$f, inds...] = getindex(rhs, s, inds, bounds)) : ())
end

# module StaggeredKernels