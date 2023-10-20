# module StaggeredKernels

export linearize, SchurComplement

struct SchurComplement # eliminates f_y(x, y) when linearized
	f_x
	f_y
	y
end

(s::SchurComplement)(x) = s.f_x(x, s.y)

function linearize(s::SchurComplement, x; h = eps(Float32))
	A = linearize(s.f_x, (    ),    x , (s.y,), h = h)
	B = linearize(s.f_x, (  x,),  s.y , (    ), h = h)
	C = linearize(s.f_y, (    ),    x , (s.y,), h = h)
	D = linearize(s.f_y, (  x,),  s.y , (    ), h = h)

	return x -> A(x) - B(1/D(1/C(x)))
end

cstep(f, h) = imag(f) / h
cstep(f::Pair{String}, h) = f[1] => cstep(f[2], h)
(cstep(f::Tuple{F, BCs},  h) where {F <: AbstractField,  BCs <: Tuple})      = (cstep(f[1], h), map(bc -> cstep(bc, h), f[2]))
(cstep(f::Tuple{F, BCs},  h) where {F <: AbstractTensor, BCs <: NamedTuple}) = (cstep(f[1], h), map(bc -> map(bc_ -> cstep(bc_, h), bc), f[2]))

linearize(f, x; h = eps(Float32)) = v -> cstep(f(x + h * im * v), h)
linearize(f, a::Tuple, x, b::Tuple; h = eps(Float32)) = v -> cstep(f(a..., x + h * im * v, b...), h)

(Base.:-(a::Tuple{A, BCs}) where {A, BCs <: Tuple}) = (-a[1], map(bc -> -bc, a[2]))
(Base.:-(bc::BC{D}) where D) = BC{D}(-bc.expr)

(Base.:+(a::Tuple{A, ABCs}, b::Tuple{B, BBCs}) where {A, B, ABCs <: Tuple, BBCs <: Tuple}) = (a[1] + b[1], a[2] .+ b[2])
(Base.:+(a::Tuple{A, BCs}, b::B) where {A <: AbstractField, B <: Scalar, BCs <: Tuple}) = (a[1] + b, map(bc -> bc + b, a[2]))
(Base.:+(a::A, b::Tuple{B, BCs}) where {A <: Scalar, B <: AbstractField, BCs <: Tuple}) = (a + b[1], map(bc -> a + bc, b[2]))
(Base.:+(a::Tuple{A, ABCs}, b::Tuple{B, BBCs}) where {A <: AbstractTensor, B <: AbstractTensor, ABCs <: NamedTuple, BBCs <: NamedTuple}) = (a[1] + b[1], (; zip(keys(a[2]), map((bca, bcb) -> bca + bcb, getproperty(a[2], k), getproperty(b[2], k)) for k in intersect(keys(a[2]), keys(b[2])))...))
(Base.:+(a::Tuple{A, BCs}, b::B) where {A <: AbstractTensor, B <: AbstractTensor, BCs <: NamedTuple}) = (a[1] + b, (; zip(keys(a[2]), map(bc -> bc + getproperty(b, k), getproperty(a[2], k)) for k in keys(a[2]))...))
(Base.:+(a::A, b::Tuple{B, BCs}) where {A <: AbstractTensor, B <: AbstractTensor, BCs <: NamedTuple}) = (a + b[1], (; zip(keys(b[2]), map(bc -> getproperty(a, k) + bc, getproperty(b[2], k)) for k in keys(b[2]))...))
(Base.:+(a::Tuple{A, BCs}, b::B) where {A <: AbstractTensor, B, BCs <: NamedTuple}) = (a[1] + b, (; zip(keys(a[2]), map(bc -> bc + b, getproperty(a[2], k)) for k in keys(a[2]))...))
(Base.:+(a::A, b::Tuple{B, BCs}) where {A, B <: AbstractTensor, BCs <: NamedTuple}) = (a + b[1], (; zip(keys(b[2]), map(bc -> a + bc, getproperty(b[2], k)) for k in keys(b[2]))...))
(Base.:+(a::Pair, b      )) = a[1] => a[2] + b
(Base.:+(a,       b::Pair)) = b[1] => a + b[2]
(Base.:+(a::Pair, b::Pair)) = a[1] == b[1] ? a[1] => a[2] + b[2] : error("Unequal keys.")

(Base.:-(a::Tuple{A, ABCs}, b::Tuple{B, BBCs}) where {A, B, ABCs <: Tuple, BBCs <: Tuple}) = (a[1] - b[1], a[2] .- b[2])
(Base.:-(a::Tuple{A, BCs}, b::B) where {A <: AbstractField, B <: Scalar, BCs <: Tuple}) = (a[1] - b, map(bc -> bc - b, a[2]))
(Base.:-(a::A, b::Tuple{B, BCs}) where {A <: Scalar, B <: AbstractField, BCs <: Tuple}) = (a - b[1], map(bc -> a - bc, b[2]))
(Base.:-(a::Tuple{A, ABCs}, b::Tuple{B, BBCs}) where {A <: AbstractTensor, B <: AbstractTensor, ABCs <: NamedTuple, BBCs <: NamedTuple}) = (a[1] - b[1], (; zip(keys(a[2]), map((bca, bcb) -> bca - bcb, getproperty(a[2], k), getproperty(b[2], k)) for k in intersect(keys(a[2]), keys(b[2])))...))
(Base.:-(a::Tuple{A, BCs}, b::B) where {A <: AbstractTensor, B <: AbstractTensor, BCs <: NamedTuple}) = (a[1] - b, (; zip(keys(a[2]), map(bc -> bc - getproperty(b, k), getproperty(a[2], k)) for k in keys(a[2]))...))
(Base.:-(a::A, b::Tuple{B, BCs}) where {A <: AbstractTensor, B <: AbstractTensor, BCs <: NamedTuple}) = (a - b[1], (; zip(keys(b[2]), map(bc -> getproperty(a, k) - bc, getproperty(b[2], k)) for k in keys(b[2]))...))
(Base.:-(a::Tuple{A, BCs}, b::B) where {A <: AbstractTensor, B, BCs <: NamedTuple}) = (a[1] - b, (; zip(keys(a[2]), map(bc -> bc - b, getproperty(a[2], k)) for k in keys(a[2]))...))
(Base.:-(a::A, b::Tuple{B, BCs}) where {A, B <: AbstractTensor, BCs <: NamedTuple}) = (a - b[1], (; zip(keys(b[2]), map(bc -> a - bc, getproperty(b[2], k)) for k in keys(b[2]))...))
(Base.:-(a::Pair, b      )) = a[1] => a[2] - b
(Base.:-(a,       b::Pair)) = b[1] => a - b[2]
(Base.:-(a::Pair, b::Pair)) = a[1] == b[1] ? a[1] => a[2] - b[2] : error("Unequal keys.")

(Base.:*(a::Tuple{A, ABCs}, b::Tuple{B, BBCs}) where {A, B, ABCs <: Tuple, BBCs <: Tuple}) = (a[1] * b[1], a[2] .* b[2])
(Base.:*(a::Tuple{A, BCs}, b::B) where {A <: AbstractField, B <: Scalar, BCs <: Tuple}) = (a[1] * b, map(bc -> bc * b, a[2]))
(Base.:*(a::A, b::Tuple{B, BCs}) where {A <: Scalar, B <: AbstractField, BCs <: Tuple}) = (a * b[1], map(bc -> a * bc, b[2]))
(Base.:*(a::Tuple{A, ABCs}, b::Tuple{B, BBCs}) where {A <: AbstractTensor, B <: AbstractTensor, ABCs <: NamedTuple, BBCs <: NamedTuple}) = (a[1] * b[1], (; zip(keys(a[2]), map((bca, bcb) -> bca * bcb, getproperty(a[2], k), getproperty(b[2], k)) for k in intersect(keys(a[2]), keys(b[2])))...))
(Base.:*(a::Tuple{A, BCs}, b::B) where {A <: AbstractTensor, B <: AbstractTensor, BCs <: NamedTuple}) = (a[1] * b, (; zip(keys(a[2]), map(bc -> bc * getproperty(b, k), getproperty(a[2], k)) for k in keys(a[2]))...))
(Base.:*(a::A, b::Tuple{B, BCs}) where {A <: AbstractTensor, B <: AbstractTensor, BCs <: NamedTuple}) = (a * b[1], (; zip(keys(b[2]), map(bc -> getproperty(a, k) * bc, getproperty(b[2], k)) for k in keys(b[2]))...))
(Base.:*(a::Tuple{A, BCs}, b::B) where {A <: AbstractTensor, B, BCs <: NamedTuple}) = (a[1] * b, (; zip(keys(a[2]), map(bc -> bc * b, getproperty(a[2], k)) for k in keys(a[2]))...))
(Base.:*(a::A, b::Tuple{B, BCs}) where {A, B <: AbstractTensor, BCs <: NamedTuple}) = (a * b[1], (; zip(keys(b[2]), map(bc -> a * bc, getproperty(b[2], k)) for k in keys(b[2]))...))
(Base.:*(a::Pair, b      )) = a[1] => a[2] * b
(Base.:*(a,       b::Pair)) = b[1] => a * b[2]
(Base.:*(a::Pair, b::Pair)) = a[1] == b[1] ? a[1] => a[2] * b[2] : error("Unequal keys.")

(Base.:/(a::Tuple{A, ABCs}, b::Tuple{B, BBCs}) where {A, B, ABCs <: Tuple, BBCs <: Tuple}) = (a[1] / b[1], a[2] ./ b[2])
(Base.:/(a::Tuple{A, BCs}, b::B) where {A <: AbstractField, B <: Scalar, BCs <: Tuple}) = (a[1] / b, map(bc -> bc / b, a[2]))
(Base.:/(a::A, b::Tuple{B, BCs}) where {A <: Scalar, B <: AbstractField, BCs <: Tuple}) = (a / b[1], map(bc -> a / bc, b[2]))
(Base.:/(a::Tuple{A, ABCs}, b::Tuple{B, BBCs}) where {A <: AbstractTensor, B <: AbstractTensor, ABCs <: NamedTuple, BBCs <: NamedTuple}) = (a[1] / b[1], (; zip(keys(a[2]), map((bca, bcb) -> bca / bcb, getproperty(a[2], k), getproperty(b[2], k)) for k in intersect(keys(a[2]), keys(b[2])))...))
(Base.:/(a::Tuple{A, BCs}, b::B) where {A <: AbstractTensor, B <: AbstractTensor, BCs <: NamedTuple}) = (a[1] / b, (; zip(keys(a[2]), map(bc -> bc / getproperty(b, k), getproperty(a[2], k)) for k in keys(a[2]))...))
(Base.:/(a::A, b::Tuple{B, BCs}) where {A <: AbstractTensor, B <: AbstractTensor, BCs <: NamedTuple}) = (a / b[1], (; zip(keys(b[2]), map(bc -> getproperty(a, k) / bc, getproperty(b[2], k)) for k in keys(b[2]))...))
(Base.:/(a::Tuple{A, BCs}, b::B) where {A <: AbstractTensor, B, BCs <: NamedTuple}) = (a[1] / b, (; zip(keys(a[2]), map(bc -> bc / b, getproperty(a[2], k)) for k in keys(a[2]))...))
(Base.:/(a::A, b::Tuple{B, BCs}) where {A, B <: AbstractTensor, BCs <: NamedTuple}) = (a / b[1], (; zip(keys(b[2]), map(bc -> a / bc, getproperty(b[2], k)) for k in keys(b[2]))...))
(Base.:/(a::Pair, b      )) = a[1] => a[2] / b
(Base.:/(a,       b::Pair)) = b[1] => a / b[2]
(Base.:/(a::Pair, b::Pair)) = a[1] == b[1] ? a[1] => a[2] / b[2] : error("Unequal keys.")

# module StaggeredKernels