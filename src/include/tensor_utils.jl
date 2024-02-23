# module StaggeredKernels/tensor.jl

# (tensor_order(::T) where T <: AbstractTensor) = tensor_order(T)

(tensor_order(::Type{Tensor{S,T}}) where {S,T}) = symmetry_order(S)

function tensor_order(::Type{TensorProd{T1, T2}}) where {T1, T2}
	oa = tensor_order(T1)
	ob = tensor_order(T2)
	
	# 2 indices are contracted iff:
	if max(oa,ob) >= 4 && min(oa,ob) >= 2
		cc = 2
	else # otherwise just one:
		cc = 1
	end
	
	return oa + ob - 2*cc
end

(tensor_order(::Type{TensorAdjoint{T}}) where T) = tensor_order(T)

function tensor_order(::Type{TensorExpr{T}}) where {T}
	heads  = expr_heads(fieldtypes(T)...)
	first  = length(heads)+1
	last   = length(fieldtypes(T))
	orders = sort <| [tensor_order(fieldtypes(T)[i]) for i in first:last]
	all(o -> o == orders[end] || o == 0, orders) || error("trying to operate on tensors of unequal ranks.")
	return orders[end]
end

(tensor_order(::Type{S}) where {S <: AbstractScalar}) = 0


(symmetry_order(::TensorSymmetry{O}) where O) = O
(symmetry_order(::Type{S}) where S <: TensorSymmetry) = symmetry_order(S())

(tensor_components(::Type{Tensor{S, NamedTuple{N,T}}}) where {S,N,T}) = N
(tensor_components(::T) where {T <: Tensor}) = tensor_components(T)

function check_indices(::Type{S}, inds) where S <: TensorSymmetry
	inds isa (NTuple{N,I} where {N, I <: Int}) ||
		error("Component $inds is not a valid combination of (1, 2, 3).")
	all(0 .< inds .< 4) ||
		error("Indices $inds out of range (0 .< inds .< 4)")
	length(inds) == symmetry_order(S()) ||
		error("Incorrect number of components specified for a tensor of given order.")
end

function encode_component(inds)
	O = length(inds)
	return encode_component(Unsymmetric{O}, inds)
end

function encode_component(::Type{S}, inds) where S <: TensorSymmetry
	check_indices(S, inds)
	c = Symbol(["x", "y", "z"][[inds...]]...)
	return c
end

function decode_component(s::Symbol)
	O = length(split(String(s), ""))
	return decode_component(Unsymmetric{O}, s)
end

function decode_component(::Type{S}, s::Symbol) where S <: TensorSymmetry
	dict = Dict(["x" => 1, "y" => 2, "z" => 3])
	keys = split(String(s), "")
	inds = Tuple([get(dict, k, "") for k in keys])
	check_indices(S, inds)
	return inds
end

function scalar_tensor_component(::Type{S}, data, f, c, stags) where S <: TensorSymmetry
	inds = decode_component(S, c)
	!symmetry_permutes(S, inds) ||
		error("Component ($c) does not map to itself under the symmetry $S.")
	return Field{stags}(view(data, f:(f-1+length(stags)), fill(:,ndims(data)-1)...))
end

# returns tensor order and empty string if named tuple could be parsed as tensor,
# returns -1 and error message otherwise.
function parse_tensor(::Type{NamedTuple{N,T}}) where {N, T}
	cpnt_names = String.(N)
	dim_counts = length.(cpnt_names)
	cpnt_chars = unique(join(cpnt_names))
	
	length(cpnt_names) > 0 ||
		return (0, "no components, may be OK.")
	
	all(dim_counts .== dim_counts[1]) ||
		return (-1, "not all components are of the same dimensionality.")
	
	issubset(cpnt_chars, ['x','y','z']) ||
		return (-1, "components other than x, y, or z found.")
	
	return (dim_counts[1], "")
end

# see previous comment
function parse_tensor(t::Type{NamedTuple{N,T}}, ::Type{S}) where {N, T, S}
	order, msg = parse_tensor(t)
	
	order == symmetry_order(S) || order == 0 ||
		return (-1, "dimensionality does not match that of the symmetry class $S.")
	
	# TODO: check compatibility of components with the specified symmetry.
	
	return (order, msg)
end

# module StaggeredKernels/tensor.jl