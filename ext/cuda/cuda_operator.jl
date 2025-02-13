# Encapsulates the CUDA backend specific data (e.g. element caches, memory allocation, etc.)
struct CudaElementAssembly{Ti <: Integer, MemAlloc,ELementsCaches,DHType<:AbstractDofHandler} <: AbstractElementAssembly
    threads::Ti
    blocks::Ti
    mem_alloc::MemAlloc
    eles_caches:: ELementsCaches
    strategy::CudaAssemblyStrategy
    dh::DHType
end

function Thunderbolt.init_linear_operator(strategy::CudaAssemblyStrategy,protocol::IntegrandType,qrc::QuadratureRuleCollection,dh::AbstractDofHandler ) where {IntegrandType}
    if CUDA.functional()
        return _init_linop_cuda(strategy,protocol,qrc,dh)
    else
        error("CUDA is not functional, please check your GPU driver and CUDA installation")
    end
end

function _init_linop_cuda(strategy::CudaAssemblyStrategy,protocol::IntegrandType,qrc::QuadratureRuleCollection,dh::AbstractDofHandler) where {IntegrandType}
    IT = inttype(strategy)
    FT = floattype(strategy)
    b = CUDA.zeros(FT, ndofs(dh))
    cu_dh = Adapt.adapt_structure(strategy, dh)
    n_cells = dh |> get_grid |> getncells |> (x -> convert(IT, x))
    threads = convert(IT, min(n_cells, 256))
    blocks = _calculate_nblocks(threads, n_cells)
    n_basefuncs = convert(IT,ndofs_per_cell(dh)) 
    eles_caches = _setup_caches(strategy,protocol,qrc,dh)
    mem_alloc = allocate_device_mem(FeMemShape{FT}, threads, n_basefuncs)
    element_assembly = CudaElementAssembly(threads, blocks, mem_alloc,eles_caches,strategy,cu_dh)
    return GeneralLinearOperator(b, element_assembly)
end


function _calculate_nblocks(threads::Ti, n_cells::Ti) where {Ti <: Integer}
    dev = device()
    no_sms = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    required_blocks = cld(n_cells, threads)
    required_blocks < 2 * no_sms || return convert(Ti, 2 * no_sms)
    return convert(Ti, required_blocks)
end


function _setup_caches(strategy::CudaAssemblyStrategy,integrand::IntegrandType,qrc::QuadratureRuleCollection,dh::AbstractDofHandler) where {IntegrandType}
    sdh_to_cache = sdh  -> 
    begin
        # Prepare evaluation caches
        ip          = Ferrite.getfieldinterpolation(sdh, sdh.field_names[1])
        element_qr  = getquadraturerule(qrc, sdh)
        
        # Build evaluation caches
        element_cache =  Adapt.adapt_structure(strategy,setup_element_cache(integrand, element_qr, ip, sdh))
        return element_cache
    end
    eles_caches  = dh.subdofhandlers .|> sdh_to_cache 
    return eles_caches
end


function _launch_kernel!(ker, threads, blocks, ::AbstractDeviceGlobalMem)
    CUDA.@cuda threads=threads blocks=blocks ker()
    return nothing
end

function _launch_kernel!(ker, threads, blocks, mem_alloc::AbstractDeviceSharedMem)
    shmem_size = mem_size(mem_alloc)
    CUDA.@sync CUDA.@cuda threads=threads blocks=blocks  shmem = shmem_size ker()
    return nothing
end

function Thunderbolt.update_operator!(op::GeneralLinearOperator{<:CudaElementAssembly}, time) 
    @unpack b, element_assembly = op
    @unpack threads, blocks, mem_alloc, eles_caches, strategy, dh = element_assembly
    partial_ker = (sdh,ele_cache) -> _update_linear_operator_kernel!(b, sdh, ele_cache,mem_alloc, time)
    for sdh_idx in 1:length(dh.subdofhandlers)
        sdh = dh.subdofhandlers[sdh_idx]
        ele_cache = eles_caches[sdh_idx]
        ker = () -> partial_ker(sdh,ele_cache)
        _launch_kernel!(ker, threads, blocks, mem_alloc)
    end
end

function _update_linear_operator_kernel!(b, sdh, element_cache,mem_alloc, time)
    for cell in CellIterator(sdh, mem_alloc)
        bₑ = cellfe(cell)
        assemble_element!(bₑ, cell, element_cache, time)
        dofs = celldofs(cell)
        @inbounds for i in 1:length(dofs)
            b[dofs[i]] += bₑ[i]
        end
    end
    return nothing
end
