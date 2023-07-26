# module StaggeredKernels

(cstep(f, h)) = imag(f) / h
(cstep(bc::Essential{D}, h) where D) = Essential{D}(cstep(bc.expr, h))
(cstep(  bc::Natural{D}, h) where D) =   Natural{D}(cstep(bc.expr, h))
(cstep(f::Tuple{F, BCs}, h) where {F, BCs <: Tuple}) = (cstep(f[1], h), map(bc -> cstep(bc, h), f[2]))

linearize(f, x; h = eps(Float32)) = v -> cstep(f(x + h * im * v), h)

(Base.:-(a::Tuple{A, BCs}) where {A, BCs <: Tuple}) = (-a[1], map(bc -> -bc, a[2]))
(Base.:-(bc::Essential{D}) where D) = Essential{D}(-bc.expr)
(Base.:-(  bc::Natural{D}) where D) =   Natural{D}(-bc.expr)

(Base.:-(a::Tuple{A, BCs}, b::Tuple{B, BCs}) where {A, B, BCs <: Tuple}) = (a[1] - b[1], a[2] .- b[2])
(Base.:-(a::Tuple{A, BCs}, b::B) where {A, B, BCs <: Tuple}) = (a[1] - b, map(bc -> bc - b, a[2]))
(Base.:-(a::A, b::Tuple{B, BCs}) where {A, B, BCs <: Tuple}) = (a - b[1], map(bc -> a - bc, b[2]))
(Base.:-(a::Essential{D}, b::B) where {D, B <: AbstractField}) = Essential{D}(a.expr - b)
(Base.:-(a::A, b::Essential{D}) where {D, A <: AbstractField}) = Essential{D}(a - b.expr)
# (Base.:-(a::Natural{D}, b::B) where {D, B <: AbstractField}) = Natural{D}(...)
# (Base.:-(a::A, b::Natural{D}) where {D, A <: AbstractField}) = Natural{D}(...)
(Base.:-(a::Essential{D}, b::Essential{D}) where D) = Essential{D}(a.expr - b.expr)
(Base.:-(  a::Natural{D},   b::Natural{D}) where D) =   Natural{D}(a.expr - b.expr)

# module StaggeredKernels