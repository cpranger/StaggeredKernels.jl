# module StaggeredKernels/tensor.jl

export Ones, Identity, MajorIdentity, MinorIdentity, Alternating, Kronecker, LeviCivita
export Transverse_x, Transverse_y, Transverse_z
export Diagonal, Orthotropic
export Symmetric, Anisotropic
export Antisymmetric, Unsymmetric

struct Ones{O}            <: TensorSymmetry{O}; end
struct Identity           <: TensorSymmetry{2}; end
struct MajorIdentity      <: TensorSymmetry{4}; end
struct MinorIdentity      <: TensorSymmetry{4}; end
struct Alternating        <: TensorSymmetry{3}; end

Identity()       =  Tensor(Identity)
MajorIdentity()  =  Tensor(MajorIdentity)
MinorIdentity()  =  Tensor(MinorIdentity)
Alternating()    =  Tensor(Alternating)

@generated function symmetry_expr(::Type{Ones}, x, ::Val{I}) where I
	J = (1, 1); return :(get_component_by_index(x, Val{$J}()))
end

function symmetry_permutes(::Type{Ones}, inds)
	return !all(inds .== (1, 1))
end

Tensor(::Type{Ones}) = Tensor((xx = 1,), Ones)

@generated function symmetry_expr(::Type{Identity}, x, ::Val{I}) where I
	i, j = I
	i == j  ||  return :(0)
	J = (1, 1); return :(get_component_by_index(x, Val{$J}()))
end

function symmetry_permutes(::Type{Identity}, inds)
	return !all(inds .== (1, 1))
end

Tensor(::Type{Identity}) = Tensor((xx = 1,), Identity)

@generated function symmetry_expr(::Type{MajorIdentity}, x, ::Val{I}) where I
	i, j, k, l = I
	i == j && k == l || return :(0)
	J = (1, 1, 1, 1);   return :(get_component_by_index(x, Val{$J}()))
end

function symmetry_permutes(::Type{MajorIdentity}, inds)
	return !all(inds .== (1, 1, 1, 1))
end

Tensor(::Type{MajorIdentity}) = Tensor((xxxx = 1,), MajorIdentity)

@generated function symmetry_expr(::Type{MinorIdentity}, x, ::Val{I}) where I
	i, j, k, l = I
	f = Int(i == k && j == l) + Int(i == l && j == k)
	f > 0 || return :(0)
	J = (1, 2, 1, 2); return :($f * get_component_by_index(x, Val{$J}()))
end

function symmetry_permutes(::Type{MinorIdentity}, inds)
	return !all(inds .== (1, 2, 1, 2))
end

Tensor(::Type{MinorIdentity}) = Tensor((xyxy = 1,), MinorIdentity)

@generated function symmetry_expr(::Type{Alternating}, x, ::Val{I}) where I
	i, j, k = I
	f = Int((i,j,k) in [(1,2,3), (2,3,1), (3,1,2)]) -
	    Int((i,j,k) in [(1,3,2), (2,1,3), (3,2,1)])
	f > 0 || return :(0)
	J = (1, 2, 3); return :($f * get_component_by_index(x, Val{$J}()))
end

function symmetry_permutes(::Type{Alternating}, inds)
	return !all(inds .== (1, 2, 3))
end

Tensor(::Type{Alternating}) = Tensor((xyz = 1,), Alternating)

const Kronecker  = Identity
const LeviCivita = Alternating

struct Transverse_x      <: TensorSymmetry{2}; end
struct Transverse_y      <: TensorSymmetry{2}; end
struct Transverse_z      <: TensorSymmetry{2}; end

@generated function symmetry_expr(::Type{Transverse_x}, x, ::Val{I}) where I
	i, j = I
	i == j  ||  return :(0)
	i == 1  &&  return :(get_component_by_index(x, Val{$I}()))
	J = (2, 2); return :(get_component_by_index(x, Val{$J}()))
end

function symmetry_permutes(::Type{Transverse_x}, inds)
	i, j = inds; return !(i == j == 1 || i == j == 2)
end

@generated function symmetry_expr(::Type{Transverse_y}, x, ::Val{I}) where I
	i, j = I
	i == j  ||  return :(0)
	i == 2  &&  return :(get_component_by_index(x, Val{$I}()))
	J = (3, 3); return :(get_component_by_index(x, Val{$J}()))
end

function symmetry_permutes(::Type{Transverse_y}, inds)
	i, j = inds; return !(i == j == 2 || i == j == 3)
end

@generated function symmetry_expr(::Type{Transverse_z}, x, ::Val{I}) where I
	i, j = I
	i == j  ||  return :(0)
	i == 3  &&  return :(get_component_by_index(x, Val{$I}()))
	J = (1, 1); return :(get_component_by_index(x, Val{$J}()))
end

function symmetry_permutes(::Type{Transverse_z}, inds)
	i, j = inds; return !(i == j == 3 || i == j == 1)
end

# Suggest creating 4-th order transversely isotropic stiffness tensor as (e.g.)
# λ = Tensor(Transverse_z(), [...])
# μ = Tensor(Transverse_z(), [...])
# C = λ * Tensor(MajorIdentity()) + μ * Tensor(MinorIdentity())

struct Diagonal           <: TensorSymmetry{2}; end
const Orthotropic = Diagonal

@generated function symmetry_expr(::Type{Diagonal}, x, ::Val{I}) where I
	i, j = I
	i == j && return :(get_component_by_index(x, Val{I}())) 
	return :(0)
end

function symmetry_permutes(::Type{Diagonal}, inds)
	i, j = inds; return !(i == j)
end
# Suggest creating 4-th order orthotropic stiffness tensor as (e.g.)
# λ = Tensor(Orthotropic(), [...])
# μ = Tensor(Orthotropic(), [...])
# C = λ * Tensor(MajorIdentity()) + μ * Tensor(MinorIdentity())

struct Symmetric           <: TensorSymmetry{2}; end
const Anisotropic = Symmetric

@generated function symmetry_expr(::Type{Symmetric}, x, ::Val{I}) where I
	i, j = I
	i <= j  &&  return :(get_component_by_index(x, Val{$I}()))
	J = (j, i); return :(get_component_by_index(x, Val{$J}()))
end

function symmetry_permutes(::Type{Symmetric}, inds)
	i, j = inds; return !(i <= j)
end
# Suggest creating 4-th order anisotropic stiffness tensor as (e.g.)
# λ = Tensor(Anisotropic(), [...])
# μ = Tensor(Anisotropic(), [...])
# C = λ * Tensor(MajorIdentity()) + μ * Tensor(MinorIdentity())

struct Antisymmetric      <: TensorSymmetry{2}; end

@generated function symmetry_expr(::Type{Antisymmetric}, x, ::Val{I}) where I
	i, j = I
	i == j  &&  return :(0)
	i <  j  &&  return :( get_component_by_index(x, Val{$I}()))
	J = (j, i); return :(-get_component_by_index(x, Val{$J}()))
end

function symmetry_permutes(::Type{Antisymmetric}, inds)
	i, j = inds; return !(i < j)
end

struct Unsymmetric{O}     <: TensorSymmetry{O}; end

@generated function symmetry_expr(::Type{Unsymmetric{O}}, x, ::Val{I}) where {O, I}
	return :(get_component_by_index(x, Val{I}()))
end

function symmetry_permutes(::Type{Unsymmetric{O}}, inds) where O
	return !true
end

# module StaggeredKernels/tensor.jl