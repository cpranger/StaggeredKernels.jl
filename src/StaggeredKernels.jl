module StaggeredKernels

<|(f, x) = f(x)

export   assign!, reduce, collect_stags, stag_subset, dot, l2, prepare_assignment

# internals
include("./include/scalar.jl")
include("./include/tensor.jl")
include("./include/arithmetic.jl")

@generated function assign_at!(lhs::L, rhs::R, inds, bounds) where {L <: Tuple, R <: Tuple}
	(ll, lr) = (length(lhs.parameters), length(rhs.parameters))
	ll == lr || error("length(lhs) != length(rhs)")
	exprs = [:(assign_at!(lhs[$i], rhs[$i], inds, bounds)) for i in 1:ll]
	return Expr(:block, exprs...)
end

# TODO: IDENTIFY WHERE NEEDED AND DELETE IF POSSIBLE
@generated function assign_at!(lhs::NamedTuple{N,L}, rhs::R, inds, bounds) where {N, L <: Tuple, R <: Tuple}
	exprs = [:(assign_at!(lhs[$i], rhs[$i], inds, bounds)) for i in eachindex(N)]
	return Expr(:block, exprs...)
end

@generated function assign_at!(lhs::NamedTuple{N,L}, rhs::NamedTuple{N,R}, inds, bounds) where {N, L <: Tuple, R <: Tuple}
	exprs = [:(assign_at!(lhs.$n, rhs.$n, inds, bounds)) for n in N]
	return Expr(:block, exprs...)
end

# TODO: IDENTIFY WHERE NEEDED AND DELETE IF POSSIBLE
@generated function assign_at!(lhs::L, rhs::Tuple{R, BC}, inds, bounds) where {L <: NamedTuple, R <: NamedTuple, BC <: NamedTuple}
	keys = intersect(L.parameters[1], R.parameters[1])
	exprs = [:(assign_at!(lhs.$k, (rhs[1].$k, rhs[2].$k), inds, bounds)) for k in keys]
	return Expr(:block, exprs...)
end

gridsize(f)              = missing
gridsize(f::Field)       = size(f.data)[2:end]
gridsize(f::Tensor)      = gridsize(f.cpnts)
gridsize(f::NamedTuple)  = gridsize(values(f))

function gridsize(f::Tuple)
	nn = skipmissing(gridsize.(f))
	length(collect(nn)) >= 1 || error("No gridded data structures found.")
	all(n -> n == nn[1], nn) || error("Data defined on unequal grids.")
	return nn[1]
end

(prepare_assignment(lhs::L, rhs::R) where {L <: Tuple, R <: Tuple}) = 
	map(prepare_assignment, lhs, rhs)

(prepare_assignment(lhs::NamedTuple{N,L}, rhs::NamedTuple{N,R}) where {N, L <: Tuple, R <: Tuple}) = 
	map(prepare_assignment, lhs, rhs)

# (prepare_assignment(lhs::L, rhs::Tuple{R, BC}) where {L <: NamedTuple, R <: NamedTuple, BC <: NamedTuple}) = 
# 	map((l, r, bc) -> prepare_assignment(l, (r, bc)), lhs, rhs[1], rhs[2])

function assign!(lhs, rhs)
	n = gridsize(lhs)
	o = n .- n .+ 1
	bounds = (o, n)
	prepared_rhs = prepare_assignment(lhs, rhs)
	for i in CartesianIndices <| map((:), bounds...)
		assign_at!(lhs, prepared_rhs, Tuple(i), bounds)
	end
end

@generated function reduce_at!(result::Ref{T}, op, field::F, inds, bounds) where {T, F <: NamedTuple}
	keys  = field.parameters[1]
	exprs = [:(reduce_at!(result, op, field.$k, inds, bounds)) for k in keys]
	return Expr(:block, exprs...)
end

@generated function reduce_at!(result::Ref{T}, op, f1::NamedTuple{N}, f2::NamedTuple{N}, inds, bounds) where {T, N}
	exprs = [:(reduce_at!(result, op, f1.$k, f2.$k, inds, bounds)) for k in N]
	return Expr(:block, exprs...)
end

function reduce(op, field; init = 0.)
	n = gridsize(field)
	o = n .- n .+ 1
	bounds   = (o, n)
	result   = Ref(init)
	for i in CartesianIndices <| map((:), bounds...)
		reduce_at!(result, op, field, Tuple(i), bounds)
	end
	return result[]
end

function reduce(op::Op, field1::F1, field2::F2; init::R = 0.) where {Op, F1, F2, R}
	n = gridsize((field1, field2))
	o = n .- n .+ 1
	bounds   = (o, n)
	result   = Ref(init)
	for i in CartesianIndices <| map((:), bounds...)
		reduce_at!(result, op, field1, field2, Tuple(i), bounds)
	end
	return result[]
end


collect_stags(stags::NamedTuple) = Tuple <| union(values(stags)...)

collect_stags(stags::Tensor) = collect_stags(stags.cpnts)

stag_subset(stags::NTuple, dims::UnitRange) = 
	Tuple <| unique <| map(f -> f[dims], stags)

stag_subset(stags::NamedTuple, cpnts::(NTuple{N,Symbol} where N), dims::UnitRange = 1:3) = 
	(; zip(cpnts, stag_subset(getfield(stags, c), dims) for c in cpnts)...)
	
stag_subset(stags::Tensor, cpnts, dims = 1:3) = stag_subset(stags.cpnts, cpnts, dims)

module Volume
	using ..StaggeredKernels
	export motion_stags, strain_stags, curl_stags, div_stags, state_stags
	
	const motion_stags = Tensor((
		x = ((0, 1, 1,),),
		y = ((1, 0, 1,),),
		z = ((1, 1, 0,),),
	))
	# const motion_stags = Tensor((
	# 	x = ((1, 0, 0,),),
	# 	y = ((0, 1, 0,),),
	# 	z = ((0, 0, 1,),),
	# ))
	# fully staggered grid in 3D. TODO: make work in 2D etc...
	# const motion_stags = Tensor((
	# 	x = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0),),
	# 	y = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0),),
	# 	z = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0),),
	# ))
	const strain_stags = symgrad(motion_stags)
	const div_t_stags  = divergence(strain_stags)
	const curl_stags   = curl(motion_stags)
	const div_stags    = divergence(motion_stags)
	const state_stags  = collect_stags(strain_stags)
	# const motion_stags = div_t_stags
end

module Plane
	using ..StaggeredKernels
	import ..Volume
	
	export motion_stags, strain_stags, curl_stags, div_stags, state_stags
	
	const  motion_stags =   stag_subset(Volume.motion_stags, (:x,  :y,),       1:2)
	const  curl_stags   =   stag_subset(Volume.curl_stags,   (:z,),            1:2)
	const  strain_stags =   stag_subset(Volume.strain_stags, (:xx, :xy, :yy,), 1:2)
	const  div_t_stags  =   stag_subset(Volume.div_t_stags,  (:x,  :y,),       1:2)
	const  div_stags    =   stag_subset(Volume.div_stags, 1:2)
	const  state_stags  = collect_stags(strain_stags)
	# TODO: Make dependent on velocity staggering
end

module Antiplane
	using ..StaggeredKernels
	import ..Volume
	
	export motion_stags, strain_stags, curl_stags, state_stags, div_stags
	
	const  motion_stags =   stag_subset(Volume.motion_stags, (:z,),       1:2)
	const  curl_stags   =   stag_subset(Volume.curl_stags,   (:x,  :y,),  1:2)
	const  strain_stags =   stag_subset(Volume.strain_stags, (:xz, :yz,), 1:2)
	const  div_t_stags  =   stag_subset(Volume.div_t_stags,  (:z,),       1:2)
	const  div_stags    =   stag_subset(Volume.div_stags,                 1:2)
	const  state_stags  = collect_stags(strain_stags)
end

module Line
	using ..StaggeredKernels
	import ..Volume
	
	export motion_stags, strain_stags, curl_stags, state_stags, div_stags
	
	const  motion_stags =   stag_subset(Volume.motion_stags, (:x,),  1:1)
	const  curl_stags   =   stag_subset(Volume.curl_stags,   (),     1:1)
	const  strain_stags =   stag_subset(Volume.strain_stags, (:xx,), 1:1)
	const  div_t_stags  =   stag_subset(Volume.div_t_stags,  (:x,),  1:1)
	const  div_stags    =   stag_subset(Volume.div_stags,            1:1)
	const  state_stags  = collect_stags(strain_stags)
end

module Antiline
	using ..StaggeredKernels
	import ..Volume
	
	export motion_stags, strain_stags, curl_stags, state_stags, div_stags
	
	const  motion_stags =   stag_subset(Volume.motion_stags, (:y,),  1:1)
	const  curl_stags   =   stag_subset(Volume.curl_stags,   (:z,),  1:1)
	const  strain_stags =   stag_subset(Volume.strain_stags, (:xy,), 1:1)
	const  div_t_stags  =   stag_subset(Volume.div_t_stags,  (:y,),  1:1)
	const  div_stags    =   stag_subset(Volume.div_stags,            1:1)
	const  state_stags  = collect_stags(strain_stags)
end

end # module StaggeredKernels
