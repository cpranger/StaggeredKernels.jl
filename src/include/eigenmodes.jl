# module StaggeredKernels

export Mode, sMode, pMode

function λ(j::Int, n, b)
	n > 2      || error("!(n > 2)")
	j in 1:n-2 || error("!(j in 1:n-2)")
	b[1] == b[2] == 0 && return 2sin(π/2 * (j-0/2)/(n-2/2))
	b[1] != b[2]      && return 2sin(π/2 * (j-1/2)/(n-3/2))
	b[1] == b[2] == 1 && return 2sin(π/2 * (j-0/2)/(n-4/2))
	return 0.
end

λ(j::Tuple, n, b) = sqrt(+((map(λ, j, n, b).^2)...))

function φ(j::Int, k, n, b)
	# Meta.@show k
	n > 2      || error("!(n > 2)")
	j in 1:n-2 || error("!(j in 1:n-2)")
	k in 0:n-1 || error("!(k in 0:n-1) [k = $k, n = $n]")
	b[1] == b[2] == 0 && return sin(π * (j-0/2)/(n-2/2) * (k    ))
	b[1] != b[2] == 0 && return cos(π * (j-1/2)/(n-3/2) * (k-1/2))
	b[1] != b[2] == 1 && return sin(π * (j-1/2)/(n-3/2) * (k    ))
	b[1] == b[2] == 1 && return cos(π * (j-0/2)/(n-4/2) * (k-1/2))
	return 0.
end

function κ(d, j::Tuple, n, b)
	d > length(j) && return 0.
	return (-1)^b[d][1] * λ(j[d], n[d], b[d])
end

φ(j::Tuple, k, n, b) = *(map(d -> φ(j[d], k[d], n[d], b[d]), 1:length(j))...)

cycle(i, d) = mod(i + d - 1, 3) + 1

f_p(c, j, n, b) = κ(cycle(c,  0), j, n, b)
f_s(c, j, n, b) = κ(cycle(c, +1), j, n, b) - κ(cycle(c, -1), j, n, b)

φ_p(c, j::Tuple, k, n, b) = f_p(c, j, n[c], b[c]) * φ(j, k, n[c], b[c])
φ_s(c, j::Tuple, k, n, b) = f_s(c, j, n[c], b[c]) * φ(j, k, n[c], b[c])


function parse_bc(bc::AbstractBC{D}) where D
	d = abs(D)
	x = sign(D)
	f = mod(-x, 3) # (-1, +1) -> (1, 2)
	v = -1
	typeof(bc) <: Essential{D} && (v = 0)
	typeof(bc) <:   Natural{D} && (v = 1)
	v < 0 && error("$(typeof(bc)) !in (Natural{$D}, Essential{$D})")
	return (d, f, v)
end

function parse_bcs(f::Field)
	result = map(_ -> fill(-1, 2), 1:my_ndims(f))
	for bc in f.bcs
		(d, f, v) = parse_bc(bc)
		result[d][f] = v
	end
	any(r -> (-1 in r), result) || return result
	error("boundary conditions incompletely specified")
end

parse_bcs(t::Tensor{Unsymmetric{1}}) = [parse_bcs(f) for f in t.cpnts]

function parse_gridsize(f::Field{SS}) where SS
	length(SS) == 1 ||
		error("eigenmodes only computable for singly staggered fields")
	return size(f.data)[2:end] .- 1 .+ SS[1]
end

parse_gridsize(t::Tensor{Unsymmetric{1}}) = [parse_gridsize(f) for f in t.cpnts]


abstract type AbstractMode end

struct Mode <: AbstractMode
	n
	b
end

struct sMode <: AbstractMode
	n
	b
end

struct pMode <: AbstractMode
	n
	b
end

Mode(f::Field) = Mode(parse_gridsize(f), parse_bcs(f))

sMode(v::Tensor{Unsymmetric{1}}) = sMode(parse_gridsize(v), parse_bcs(v))
pMode(v::Tensor{Unsymmetric{1}}) = pMode(parse_gridsize(v), parse_bcs(v))

common_n(n) = map(min, n...)

function common_bc_s(b)
	c = deepcopy(b)
	for i in eachindex(b)
		for j in 1:length(b[i])
			i != j && (c[i][j][1:2] = 1 .- c[i][j][1:2])
		end
	end
	all(d -> d == c[1], c) || error("error")
	return c[1]
end

function common_bc_p(b)
	c = deepcopy(b)
	for i in eachindex(b)
		for j in 1:length(b[i])
			i == j && (c[i][j][1:2] = 1 .- c[i][j][1:2])
		end
	end
	all(d -> d == c[1], c) || error("error")
	return c[1]
end

lindx(inds) = ceil.(Int, inds)

function Base.getindex(u::Mode, j_...)
	j = map((j, s) -> j >= 0 ? j : (s - 2 + 1 + j), j_, u.n)
	return (
		val = λ(j, u.n, u.b),
		gen = fieldgen((k...) -> φ(j, lindx(k), u.n, u.b))
	)
end

function Base.getindex(u::pMode, j_...)
	j = map((j, s) -> j >= 0 ? j : (s - 2 + 1 + j), j_, common_n(u.n))
	return (
		val = λ(j, common_n(u.n), common_bc_p(u.b)),
		gen = Tensor((
			x = fieldgen((k...) -> φ_p(1, j, lindx(k), u.n, u.b)),
			y = fieldgen((k...) -> φ_p(2, j, lindx(k), u.n, u.b)),
			z = fieldgen((k...) -> φ_p(3, j, lindx(k), u.n, u.b)),
		), Unsymmetric{1})
	)
end

function Base.getindex(u::sMode, j_...)
	j = map((j, s) -> j >= 0 ? j : (s - 2 + 1 + j), j_, common_n(u.n))
	return (
		val = λ(j, common_n(u.n), common_bc_s(u.b)),
		gen = Tensor((
			x = fieldgen((k...) -> φ_s(1, j, lindx(k), u.n, u.b)),
			y = fieldgen((k...) -> φ_s(2, j, lindx(k), u.n, u.b)),
			z = fieldgen((k...) -> φ_s(3, j, lindx(k), u.n, u.b)),
		), Unsymmetric{1})
	)
end

# module StaggeredKernels