# module StaggeredKernels

export If, A, D, interpolate, fieldgen, Field, Essential, Natural, Neumann, Dirichlet, FieldRed#, assign_at, reduce_at

abstract type AbstractField end
abstract type AbstractBC{D} end

const Scalar = Union{Number,AbstractField}

struct Field{Indx, BCs <: Tuple, T <: AbstractArray} <: AbstractField
	data::T
	bcs::BCs
end

(Field(dims::NTuple{N,I}, stags::NTuple{M, NTuple{N,I}}, bcs::Tuple = ()) where {M, N, I  <: Int}) =
    Field{stags}(@zeros(M, dims...), bcs)
	
(Field{SS}(data::T, bcs::BCs = ()) where {SS, BCs, T <: AbstractArray}) =
    Field{SS, BCs, T}(data, bcs)

# TODO: TEST DEEP VS SHALLOW COPY WHEN CONSTRUCTING
struct FieldExpr{T <: Tuple} <: AbstractField
	contents::T
end

(FieldExpr(args::T) where T <: Tuple) = FieldExpr{T}(args)
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
A(x::Number, ::Symbol) = x

D(x, d::Symbol) = D(x, decode_component(d)[1])
A(x, d::Symbol) = A(x, decode_component(d)[1])

D(x, d::Int   ) = d <= my_ndims(x) ? D(x, kronecker(d)(my_ndims(x))) : 0
A(x, d::Int   ) = d <= my_ndims(x) ? A(x, kronecker(d)(my_ndims(x))) : 0

D(x::AbstractField, d::NTuple) = (S(x, .+d) - S(x, .-d))
A(x::AbstractField, d::NTuple) = (S(x, .+d) + S(x, .-d))/2

(D(x::NTuple{M,NTuple{N,I}}, d::NTuple) where {N, M, I <: Int}) = Tuple <| unique <| map(s -> mod.(s .+ d, 2), x)
(A(x::NTuple{M,NTuple{N,I}}, d::NTuple) where {N, M, I <: Int}) = Tuple <| unique <| map(s -> mod.(s .+ d, 2), x)


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


struct Natural{D, T <: Scalar} <: AbstractBC{D}
	expr::T
end

struct Essential{D, T <: Scalar} <: AbstractBC{D}
	expr::T
end

const Neumann = Natural
const Dirichlet = Essential

function decode_bc_dir(sign::Symbol, axis::Symbol)
	sign_dict = Dict([:- => -1, :+ => +1])
	axis_dict = Dict([:x =>  1, :y =>  2, :z =>  3])
	
	sign in keys(sign_dict) || error("first  argument should be in $(keys(sign_dict)).")
	axis in keys(axis_dict) || error("second argument should be in $(keys(axis_dict)).")
	
	return sign_dict[sign] * axis_dict[axis]
end

(  Natural(sign::Symbol, axis::Symbol, val::T) where T <: Scalar) = 
	  Natural{decode_bc_dir(sign, axis), T}(val)

(Essential(sign::Symbol, axis::Symbol, val::T) where T <: Scalar) = 
	Essential{decode_bc_dir(sign, axis), T}(val)

(lmargin(::Type{BC} where BC <: AbstractBC{D}) where D) = D < 0 ? .-kronecker(abs(D))(3) : (0, 0, 0,)
(umargin(::Type{BC} where BC <: AbstractBC{D}) where D) = D > 0 ? .+kronecker(abs(D))(3) : (0, 0, 0,)

lmargin() = (0, 0, 0,)
umargin() = (0, 0, 0,)

lmargin(bc1, bc2, bcs...) = map(min, lmargin(bc1), lmargin(bc2, bcs...))
umargin(bc1, bc2, bcs...) = map(max, umargin(bc1), umargin(bc2, bcs...))

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

(stags(::Type{Field{St, BCs, T}}) where {St, BCs, T}) = St
(stags(::Type{FieldExpr{Tt}})     where {Tt        }) = intersect([stags(t) for t in fieldtypes(Tt)]...)
(stags(::Type{FieldShft{S,T}})    where {S,  T     }) = [mod.(s .+ S, 2) for s in stags(T)]
(stags(::Type{FieldIntp{A}})      where {A         }) = combinations([0, 1], length(stags(A)[1]))
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

getindex(x::Number, ::Val, inds...) = x

function checkbounds(x::Field{SS}, f, inds...) where SS
	Base.checkbounds(Bool, x.data, f, (inds .+ 1 .- SS[f])...) || error("Out of bounds at index $inds")
	return true
end

function Base.getindex(x::Field, inds...)
	@boundscheck checkbounds(x, inds...)
	return x.data[inds...]
end

@generated function getindex(x::Field{SS}, ::Val{S}, inds...) where {SS, S}
	f = stagindex(SS, mod.(S, 2))
	offset = stencil_offset(S)
	return :(x[$f, (inds .+ $offset)...])
end

(expr_heads(args...)) = []
(expr_heads(::Type{Val{S}}, args...) where S) = [S, expr_heads(args...)...]

@generated function getindex(x::FieldExpr{T}, s::Val{S}, inds...) where {T <: Tuple, S}
	heads = expr_heads(fieldtypes(T)...)
	first = length(heads)+1
	last  = length(fieldtypes(T))
	args  = [:(getindex(x.contents[$i], s, inds...)) for i in first:last]
	return Expr(heads..., args...)
end

(getindex(x::FieldShft{S1,T}, s::Val{S2}, inds...) where {T, S1, S2}) = 
	getindex(x.shiftee, Val(S1 .+ S2), inds...)

(getindex(x::FieldGen, s::Val{S}, inds...) where S) = x.func((inds .+ S./2 .- 1)...)

@generated function getindex(x::FieldIntp{A}, s::Val{S}, inds...) where {A, S}
	sten = stencil(FieldIntp{A}, S)
	size = length(sten)
	args = [:(getindex(x.interpolant, $(Val(s)), inds...)) for s in sten]
	return Expr(:call, :/, Expr(:call, :+, args...), size)
end


@generated function reduce_at!(result::AbstractArray{T,0}, op, f::Field{S}, inds, bounds) where {T, S}
	return Expr(
		:block,
		[Expr(
			:if,
			:(all(bounds[1] .<= inds .<= bounds[2] .+ $(S[i] .- 1))),
			:(result[] = op(result[], f.data[$i, inds...]))
		) for i in eachindex(S)]...
	)
end

@generated function reduce_at!(result::AbstractArray{T,0}, op, f1::Field{S}, f2::Field{S}, inds, bounds) where {T, S}
	return Expr(
		:block,
		[Expr(
			:if,
			:(all(bounds[1] .<= inds .<= bounds[2] .+ $(S[i] .- 1))),
			:(result[] = op(result[], f1.data[$i, inds...], f2.data[$i, inds...]))
		) for i in eachindex(S)]...
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
			(bounds[1], bounds[2] .+ $(s .- 1))
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

@generated function assign_at!(lhs::Field{Stags, BCs}, rhs, s::Val{Stag}, inds, bounds) where {Stags, BCs, Stag}
	n = length(Stag)
	i = stagindex(Stags, mod.(Stag, 2))
	
	lm = lmargin(fieldtypes(BCs)...)[1:n] # should be negative
	um = umargin(fieldtypes(BCs)...)[1:n] # should be positive
	
	margin = (um .- lm)

	# rotate indices counter-clockwise so that boundary conditions are computed last :-D
	test_expr =  :(all(bounds[1] .<= inds .<= bounds[2] .- $margin))
	then_expr =  :(lhs.data[$i, (inds .- $lm)...] = getindex(rhs, s, (inds .- $lm)...))
	bc_exprs  = [:(assign_bc_at!(lhs, lhs.bcs[$j], s, $(Val(i)), mod.(inds .- $lm .- 1, bounds[2]) .+ 1, bounds)) for j in 1:fieldcount(BCs)]
	else_expr =  Expr(:block, bc_exprs...)
	
	return Expr(:if, test_expr, then_expr, else_expr)
end

@generated function assign_bc_at!(lhs::Field, bc::Essential{D}, s::Val{S}, i::Val{I}, inds, bounds) where {D, I, S}
	d = abs(D)
	x = sign(D)
	f = mod(-x, 3) # (-1, +1) -> (1, 2)
	return quote
		# println("-> $inds | $(bounds[2])")
		if inds[$d] == bounds[$f][$d]
			# println("Natural($D): $inds")
			lhs.data[$I, inds...] = getindex(bc.expr, s, inds...)
		end
	end
end

@generated function assign_bc_at!(lhs::Field, bc::Natural{D}, s::Val{S}, i::Val{I}, inds, bounds) where {D, I, S}
	d = abs(D)
	x = sign(D)
	f = mod(-x, 3) # (-1, +1) -> (1, 2)
	n = length(S)
	o = -x .* kronecker(d)(n)
	return quote
		# println("-> $inds | $(bounds[2])")
		if inds[$d] == bounds[$f][$d]
			# println("Natural($D): $inds")
			lhs.data[$I, inds...] = lhs.data[$I, (inds .+ $o)...] + getindex(bc.expr, s, inds...)
		end
	end
end

Base.:(+)(a::AbstractField) = a
Base.:(-)(a::AbstractField) = FieldOp(:-, a)

Base.sqrt(a::AbstractField) = FieldOp(:sqrt, a)
Base.imag(a::AbstractField) = FieldOp(:imag, a)
Base.real(a::AbstractField) = FieldOp(:real, a)

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

dot(a::AbstractField, b::AbstractField) = interpolate(a) * interpolate(b)

# module StaggeredKernels