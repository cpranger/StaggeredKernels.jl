# module StaggeredKernels

# export linearize, SchurComplement

# struct SchurComplement # eliminates f(x, y)[2] when linearized
# 	f
# 	y
# end

# (s::SchurComplement)(x) = s.f(x, s.y)[1]

# function linearize(s::SchurComplement, x; h = eps(Float32))
# 	f_x = (x, y) -> s.f(x, y)[1]
# 	f_y = (x, y) -> s.f(x, y)[2]
	
# 	A = linearize(f_x, (    ),    x , (s.y,), h = h)
# 	B = linearize(f_x, (  x,),  s.y , (    ), h = h)
# 	C = linearize(f_y, (    ),    x , (s.y,), h = h)
# 	D = linearize(f_y, (  x,),  s.y , (    ), h = h)

# 	return x -> A(x) - B(1/D(1/C(x)))
# end

#  cstep(f, h) = imag(f) / h
# (cstep(f::BC{D}, h) where D)    = BC{D}(cstep(f.expr, h))
#  cstep(f::AbstractScalar_BC, h) = (cstep(f[1], h), map(bc -> cstep(bc, h), f[2]))
#  cstep(f::AbstractTensor_BC, h) = (cstep(f[1], h), map(bc -> map(bc_ -> cstep(bc_, h), bc), f[2]))

# linearize(f, x; h = eps(Float32)) = v -> imag(f(x + h * im * v)) / h
# linearize(f, a::Tuple, x, b::Tuple; h = eps(Float32)) = v -> imag(f(a..., x + h * im * v, b...)) / h


# abstract type AbstractTensor end

# const ScalarBCs =               Tuple{T, Vararg{TT}}  where {   T <: BC, TT <: BC}
# const TensorBCs = NamedTuple{N, Tuple{T, Vararg{TT}}} where {N, T <: BC, TT <: BC}
# const BCs       = Union{ScalarBCs, TensorBCs}

# const AbstractScalar_BC = Tuple{A, BCs} where {A <: AbstractScalar, BCs <: ScalarBC}
# const AbstractTensor_BC = Tuple{A, BCs} where {A <: AbstractTensor, BCs <: TensorBC}

const AbstractObject    = Union{AbstractScalar, AbstractTensor}
# const AbstractObject_BC = Union{AbstractScalar_BC, AbstractTensor_BC}

# const AbstractStagObject = Union{
# 	Union{AbstractScalar, AbstractScalar_BC},
# 	Union{AbstractTensor, AbstractTensor_BC}
# }

(Base.:+(a::BC{D}) where D)         = a
#  Base.:+(a::BCs)                    = a
 Base.:+(a::AbstractObject)         = a
 Base.:+(a::Tuple)                  = a
 Base.:+(a::NamedTuple)             = a

(Base.:-(a::BC{D}) where D)         = BC{D}(-a.expr)
#  Base.:-(a::BCs)                    = map(a -> -a, a)
 Base.:-(a::AbstractScalarField)    = ScalarOp(:-, a)
 Base.:-(a::AbstractTensor)         = TensorOp(:-, a)
 Base.:-(a::Tuple)                  = .- a
 Base.:-(a::NamedTuple)             = map(a -> -a, a)

(Base.:imag(a::BC{D}) where D)      = BC{D}(imag(a.expr))
#  Base.:imag(a::BCs)                 = map(a -> imag(a), a)
 Base.:imag(a::AbstractScalarField) = ScalarOp(:imag, a)
 Base.:imag(a::AbstractTensor)      = TensorOp(:imag, a)
 Base.:imag(a::Tuple)               = imag.(a)
 Base.:imag(a::NamedTuple)          = map(a -> imag(a), a)

(Base.:real(a::BC{D}) where D)      = BC{D}(real(a.expr))
#  Base.:real(a::BCs)                 = map(a -> real(a), a)
 Base.:real(a::AbstractScalarField) = ScalarOp(:real, a)
 Base.:real(a::AbstractTensor)      = TensorOp(:real, a)
 Base.:real(a::Tuple)               = real.(a)
 Base.:real(a::NamedTuple)          = map(a -> real(a), a)

(Base.:conj(a::BC{D}) where D)      = BC{D}(conj(a.expr))
#  Base.:conj(a::BCs)                 = map(a -> conj(a), a)
 Base.:conj(a::AbstractScalarField) = ScalarOp(:conj, a)
 Base.:conj(a::AbstractTensor)      = TensorOp(:conj, a)
 Base.:conj(a::Tuple)               = conj.(a)
 Base.:conj(a::NamedTuple)          = map(a -> conj(a), a)

(Base.:abs(a::BC{D}) where D)       = BC{D}(abs(a.expr))
#  Base.:abs(a::BCs)                  = map(a -> abs(a), a)
 Base.:abs(a::AbstractScalarField)  = ScalarOp(:abs, a)
 Base.:abs(a::AbstractTensor)       = TensorOp(:abs, a)
 Base.:abs(a::Tuple)                = abs.(a)
 Base.:abs(a::NamedTuple)           = map(a -> abs(a), a)

(Base.:exp(a::BC{D}) where D)       = BC{D}(exp(a.expr))
#  Base.:exp(a::BCs)                  = map(a -> exp(a), a)
 Base.:exp(a::AbstractScalarField)  = ScalarOp(:exp, a)
 Base.:exp(a::Tuple)                = exp.(a)
 Base.:exp(a::NamedTuple)           = map(a -> exp(a), a)

(Base.:sin(a::BC{D}) where D)       = BC{D}(sin(a.expr))
#  Base.:sin(a::BCs)                  = map(a -> sin(a), a)
 Base.:sin(a::AbstractScalarField)  = ScalarOp(:sin, a)
 Base.:sin(a::Tuple)                = sin.(a)
 Base.:sin(a::NamedTuple)           = map(a -> sin(a), a)

(Base.:cos(a::BC{D}) where D)       = BC{D}(cos(a.expr))
#  Base.:cos(a::BCs)                  = map(a -> cos(a), a)
 Base.:cos(a::AbstractScalarField)  = ScalarOp(:cos, a)
 Base.:cos(a::Tuple)                = cos.(a)
 Base.:cos(a::NamedTuple)           = map(a -> cos(a), a)

Base.length(a::AbstractObject) = 1

(Base.:+(a::BC{D},               b::BC{D}              ) where D)  =  BC{D}(a.expr + b.expr)
(Base.:+(a::BC{D},               b::AbstractScalar     ) where D)  =  BC{D}(a.expr + b     )
(Base.:+(a::AbstractScalar,      b::BC{D}              ) where D)  =  BC{D}(a      + b.expr)
#  Base.:+(a::BCs,                 b::BCs                )           =  map((a, b) -> a + b, a, b)
#  Base.:+(a::BCs,                 b::AbstractObject     )           =  map( a     -> a + b, a   )
#  Base.:+(a::AbstractObject,      b::BCs                )           =  map(    b  -> a + b,    b)
 Base.:+(a::AbstractScalarField, b::AbstractScalarField)           =  ScalarOp(:+, a, b)
 Base.:+(a::AbstractScalar,      b::AbstractScalarField)           =  ScalarOp(:+, a, b)
 Base.:+(a::AbstractScalarField, b::AbstractScalar     )           =  ScalarOp(:+, a, b)
 Base.:+(a::AbstractTensor,      b::AbstractTensor     )           =  TensorOp(:+, a, b)
 Base.:+(a::AbstractTensor,      b::NamedTuple         )           =  TensorOp(:+, a, b)
 Base.:+(a::NamedTuple,          b::AbstractTensor     )           =  TensorOp(:+, a, b)
 Base.:+(a::NamedTuple,          b::NamedTuple         )           =  TensorOp(:+, a, b)
 Base.:+(a::AbstractObject,      b::Tuple              )           =  map(b -> a + b, b)
 Base.:+(a::Tuple,               b::AbstractObject     )           =  map(a -> a + b, a)
 Base.:+(a::Tuple,               b::Tuple              )           =  a .+ b

(Base.:-(a::BC{D},               b::BC{D}              ) where D)  =  BC{D}(a.expr - b.expr)
(Base.:-(a::BC{D},               b::AbstractScalar     ) where D)  =  BC{D}(a.expr - b     )
(Base.:-(a::AbstractScalar,      b::BC{D}              ) where D)  =  BC{D}(a      - b.expr)
#  Base.:-(a::BCs,                 b::BCs                )           =  map((a, b) -> a - b, a, b)
#  Base.:-(a::BCs,                 b::AbstractObject     )           =  map( a     -> a - b, a   )
#  Base.:-(a::AbstractObject,      b::BCs                )           =  map(    b  -> a - b,    b)
 Base.:-(a::AbstractScalarField, b::AbstractScalarField)           =  ScalarOp(:-, a, b)
 Base.:-(a::AbstractScalar,      b::AbstractScalarField)           =  ScalarOp(:-, a, b)
 Base.:-(a::AbstractScalarField, b::AbstractScalar     )           =  ScalarOp(:-, a, b)
 Base.:-(a::AbstractTensor,      b::AbstractTensor     )           =  TensorOp(:-, a, b)
 Base.:-(a::AbstractTensor,      b::NamedTuple         )           =  TensorOp(:-, a, b)
 Base.:-(a::NamedTuple,          b::AbstractTensor     )           =  TensorOp(:-, a, b)
 Base.:-(a::NamedTuple,          b::NamedTuple         )           =  TensorOp(:-, a, b)
 Base.:-(a::AbstractObject,      b::Tuple              )           =  map(b -> a - b, b)
 Base.:-(a::Tuple,               b::AbstractObject     )           =  map(a -> a - b, a)
 Base.:-(a::Tuple,               b::Tuple              )           =  a .- b

(Base.:*(a::BC{D},               b::BC{D}              ) where D)  =  BC{D}(a.expr * b.expr)
(Base.:*(a::BC{D},               b::AbstractScalar     ) where D)  =  BC{D}(a.expr * b     )
(Base.:*(a::AbstractScalar,      b::BC{D}              ) where D)  =  BC{D}(a      * b.expr)
#  Base.:*(a::ScalarBCs,           b::ScalarBCs          )           =  map((a, b) -> a * b, a, b)
#  Base.:*(a::ScalarBCs,           b::AbstractScalar     )           =  map( a     -> a * b, a   )
#  Base.:*(a::AbstractScalar,      b::ScalarBCs          )           =  map(    b  -> a * b,    b)
 Base.:*(a::AbstractScalarField, b::AbstractScalarField)           =  ScalarOp(:*, a, b)
 Base.:*(a::AbstractScalar,      b::AbstractScalarField)           =  ScalarOp(:*, a, b)
 Base.:*(a::AbstractScalarField, b::AbstractScalar     )           =  ScalarOp(:*, a, b)
 Base.:*(a::AbstractTensor,      b::AbstractTensor     )           =  TensorProd(a,   b)
 Base.:*(a::AbstractTensor,      b::AbstractScalar     )           =  TensorOp(:*, a, b)
 Base.:*(a::AbstractScalar,      b::AbstractTensor     )           =  TensorOp(:*, a, b)
 Base.:*(a::AbstractObject,      b::Tuple              )           =  map(b -> a * b, b)
 Base.:*(a::Tuple,               b::AbstractObject     )           =  map(a -> a * b, a)
 Base.:*(a::Tuple,               b::Tuple              )           =  a .* b

(Base.:/(a::BC{D},               b::BC{D}              ) where D)  =  BC{D}(a.expr / b.expr)
(Base.:/(a::BC{D},               b::AbstractScalar     ) where D)  =  BC{D}(a.expr / b     )
(Base.:/(a::AbstractScalar,      b::BC{D}              ) where D)  =  BC{D}(a      / b.expr)
#  Base.:/(a::ScalarBCs,           b::ScalarBCs          )           =  map((a, b) -> a / b, a, b)
#  Base.:/(a::ScalarBCs,           b::AbstractScalar     )           =  map( a     -> a / b, a   )
#  Base.:/(a::AbstractScalar,      b::ScalarBCs          )           =  map(    b  -> a / b,    b)
 Base.:/(a::AbstractScalarField, b::AbstractScalarField)           =  ScalarOp(:/, a, b)
 Base.:/(a::AbstractScalar,      b::AbstractScalarField)           =  ScalarOp(:/, a, b)
 Base.:/(a::AbstractScalarField, b::AbstractScalar     )           =  ScalarOp(:/, a, b)
 Base.:/(a::AbstractTensor,      b::Tensor{Unsymmetric{1}})        =  Vector((x = a.x/b.x, y = a.y/b.y, z = a.z/b.z,))
 Base.:/(a::AbstractTensor,      b::AbstractScalar     )           =  TensorOp(:/, a, b)
 Base.:/(a::AbstractScalar,      b::AbstractTensor     )           =  TensorOp(:/, a, b)
 Base.:/(a::AbstractObject,      b::Tuple              )           =  map(b -> a / b, b)
 Base.:/(a::Tuple,               b::AbstractObject     )           =  map(a -> a / b, a)
 Base.:/(a::Tuple,               b::Tuple              )           =  a ./ b

Base.:(^)( a::AbstractScalarField, b::AbstractScalarField) = ScalarOp(:^, a, b)
Base.:(^)( a::AbstractScalar     , b::AbstractScalarField) = ScalarOp(:^, a, b)
Base.:(^)( a::AbstractScalarField, b::AbstractScalar     ) = ScalarOp(:^, a, b)

Base.:(<)( a::AbstractScalarField, b::AbstractScalarField) = ScalarOp(:(<), a, b)
Base.:(<)( a::AbstractScalar     , b::AbstractScalarField) = ScalarOp(:(<), a, b)
Base.:(<)( a::AbstractScalarField, b::AbstractScalar     ) = ScalarOp(:(<), a, b)

Base.:(<=)(a::AbstractScalarField, b::AbstractScalarField) = ScalarOp(:(<=), a, b)
Base.:(<=)(a::AbstractScalar     , b::AbstractScalarField) = ScalarOp(:(<=), a, b)
Base.:(<=)(a::AbstractScalarField, b::AbstractScalar     ) = ScalarOp(:(<=), a, b)

Base.:(>)( a::AbstractScalarField, b::AbstractScalarField) = ScalarOp(:(>), a, b)
Base.:(>)( a::AbstractScalar     , b::AbstractScalarField) = ScalarOp(:(>), a, b)
Base.:(>)( a::AbstractScalarField, b::AbstractScalar     ) = ScalarOp(:(>), a, b)

Base.:(>=)(a::AbstractScalarField, b::AbstractScalarField) = ScalarOp(:(>=), a, b)
Base.:(>=)(a::AbstractScalar     , b::AbstractScalarField) = ScalarOp(:(>=), a, b)
Base.:(>=)(a::AbstractScalarField, b::AbstractScalar     ) = ScalarOp(:(>=), a, b)

Base.:(==)(a::AbstractScalarField, b::AbstractScalarField) = ScalarOp(:(==), a, b)
Base.:(==)(a::AbstractScalar     , b::AbstractScalarField) = ScalarOp(:(==), a, b)
Base.:(==)(a::AbstractScalarField, b::AbstractScalar     ) = ScalarOp(:(==), a, b)

dot(f1, f2)  =  reduce((r, a, b) -> r + a*b, f1, f2)
l2(f) = sqrt <| reduce((r, a) -> r + a^2, f)

Base.min(   f::Union{AbstractScalarField, AbstractTensor}) = reduce(min, f; init =  Inf64)
Base.max(   f::Union{AbstractScalarField, AbstractTensor}) = reduce(max, f; init = -Inf64)
Base.minmax(f::Union{AbstractScalarField, AbstractTensor}) = reduce(((mi, ma), x) -> (min(mi, x), max(ma, x)), f; init = (Inf64, -Inf64))

# module StaggeredKernels