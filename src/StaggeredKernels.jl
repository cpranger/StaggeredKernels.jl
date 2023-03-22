module StaggeredKernels

<|(f, x) = f(x)

# import ..Float
# using  ..ParallelStencil
# using    LinearAlgebra

export   assign!, reduce, collect_stags, stag_subset, dot

# internals
include("./include/scalar.jl")
include("./include/tensor.jl")
include("./include/eigenmodes.jl")

@generated function assign_at!(lhs::L, rhs::R, inds, bounds) where {L <: NamedTuple, R <: NamedTuple}
	keys = intersect(lhs.parameters[1], rhs.parameters[1])
	exprs = [:(assign_at!(lhs.$k, rhs.$k, inds, bounds)) for k in keys]
	return Expr(:block, exprs...)
end

function assign!(lhs, rhs, bounds)
	for i in CartesianIndices <| map((:), bounds...)
		assign_at!(lhs, rhs, Tuple(i), bounds)
	end
end

@generated function reduce_at!(result::AbstractArray{T,0}, op, field::F, inds, bounds) where {T, F <: NamedTuple}
	keys  = field.parameters[1]
	exprs = [:(reduce_at!(result, op, field.$k, inds, bounds)) for k in keys]
	return Expr(:block, exprs...)
end

function reduce(op, field, bounds; init = 0.)
	result   = zeros(); result[] = init
	for i in CartesianIndices <| map((:), bounds...)
		reduce_at!(result, op, field, Tuple(i), bounds)
	end
	return result[]
end

function reduce(op, field1, field2, bounds; init = 0.)
	result   = zeros(); result[] = init
	for i in CartesianIndices <| map((:), bounds...)
		reduce_at!(result, op, field1, field2, Tuple(i), bounds)
	end
	return result[]
end

dot(f1, f2, bounds) = reduce((r, a, b) -> r + a*b, f1, f2, bounds)


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
	
	const motion_stags = Vector((
		x = ((0, 1, 1,),),
		y = ((1, 0, 1,),),
		z = ((1, 1, 0,),),
	))
	# const motion_stags = Vector((
	# 	x = ((1, 0, 0,),),
	# 	y = ((0, 1, 0,),),
	# 	z = ((0, 0, 1,),),
	# ))
	# fully staggered grid in 3D. TODO: make work in 2D etc...
	# const motion_stags = Vector((
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
	
	import ..Essential, ..Natural
	export Essential, Natural, ImpermeableFreeSlip, PermeableNoSlip
	
	Essential(d::Symbol) = (Essential(:-, d, 0), Essential(:+, d, 0))
	  Natural(d::Symbol) = (  Natural(:-, d, 0),   Natural(:+, d, 0))

	Essential() = (Essential(:x)..., Essential(:y)...)
	  Natural() = (  Natural(:x)...,   Natural(:y)...)

	ImpermeableFreeSlip() = (
		s = Essential(),
		p = Natural(),
		v = (
			x = (Essential(:x)..., Natural(:y)...),
			y = (Essential(:y)..., Natural(:x)...),
		),
	)
	
	PermeableNoSlip() = (
		s = Natural(),
		p = Essential(),
		v = (
			x = (Essential(:y)..., Natural(:x)...),
			y = (Essential(:x)..., Natural(:y)...),
		),
	)
	
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
