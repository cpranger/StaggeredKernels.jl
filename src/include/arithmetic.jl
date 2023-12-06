# module StaggeredKernels

export AbstractObject

const AbstractObject    = Union{AbstractScalar, AbstractTensor}

(Base.:+(a::BC{D}) where D)         = a
 Base.:+(a::AbstractObject)         = a
 Base.:+(a::Tuple)                  = a
 Base.:+(a::NamedTuple)             = a

(Base.:-(a::BC{D}) where D)         = BC{D}(-a.expr)
 Base.:-(a::AbstractScalarField)    = ScalarOp(:-, a)
 Base.:-(a::AbstractTensor)         = TensorOp(:-, a)
 Base.:-(a::Tuple)                  = .- a
 Base.:-(a::NamedTuple)             = map(a -> -a, a)

(Base.:imag(a::BC{D}) where D)      = BC{D}(imag(a.expr))
 Base.:imag(a::AbstractScalarField) = ScalarOp(:imag, a)
 Base.:imag(a::AbstractTensor)      = TensorOp(:imag, a)
 Base.:imag(a::Tuple)               = imag.(a)
 Base.:imag(a::NamedTuple)          = map(a -> imag(a), a)

(Base.:real(a::BC{D}) where D)      = BC{D}(real(a.expr))
 Base.:real(a::AbstractScalarField) = ScalarOp(:real, a)
 Base.:real(a::AbstractTensor)      = TensorOp(:real, a)
 Base.:real(a::Tuple)               = real.(a)
 Base.:real(a::NamedTuple)          = map(a -> real(a), a)

(Base.:conj(a::BC{D}) where D)      = BC{D}(conj(a.expr))
 Base.:conj(a::AbstractScalarField) = ScalarOp(:conj, a)
 Base.:conj(a::AbstractTensor)      = TensorOp(:conj, a)
 Base.:conj(a::Tuple)               = conj.(a)
 Base.:conj(a::NamedTuple)          = map(a -> conj(a), a)

(Base.:abs(a::BC{D}) where D)       = BC{D}(abs(a.expr))
 Base.:abs(a::AbstractScalarField)  = ScalarOp(:abs, a)
 Base.:abs(a::AbstractTensor)       = TensorOp(:abs, a)
 Base.:abs(a::Tuple)                = abs.(a)
 Base.:abs(a::NamedTuple)           = map(a -> abs(a), a)

(Base.:exp(a::BC{D}) where D)       = BC{D}(exp(a.expr))
 Base.:exp(a::AbstractScalarField)  = ScalarOp(:exp, a)
 Base.:exp(a::Tuple)                = exp.(a)
 Base.:exp(a::NamedTuple)           = map(a -> exp(a), a)

(Base.:sin(a::BC{D}) where D)       = BC{D}(sin(a.expr))
 Base.:sin(a::AbstractScalarField)  = ScalarOp(:sin, a)
 Base.:sin(a::Tuple)                = sin.(a)
 Base.:sin(a::NamedTuple)           = map(a -> sin(a), a)

(Base.:cos(a::BC{D}) where D)       = BC{D}(cos(a.expr))
 Base.:cos(a::AbstractScalarField)  = ScalarOp(:cos, a)
 Base.:cos(a::Tuple)                = cos.(a)
 Base.:cos(a::NamedTuple)           = map(a -> cos(a), a)

(Base.:+(a::BC{D},               b::BC{D}              ) where D)  =  BC{D}(a.expr + b.expr)
(Base.:+(a::BC{D},               b::AbstractScalar     ) where D)  =  BC{D}(a.expr + b     )
(Base.:+(a::AbstractScalar,      b::BC{D}              ) where D)  =  BC{D}(a      + b.expr)
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