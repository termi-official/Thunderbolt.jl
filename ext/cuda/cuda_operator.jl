
# CUDA backend concrete implementation #
struct CudaOperatorKernel{Operator, Ti <: Integer, MemAlloc,ELementsCaches,DHType<:AbstractDofHandler} <: AbstractOperatorKernel{CUDABackend} 
    op::Operator
    threads::Ti
    blocks::Ti
    mem_alloc::MemAlloc
    eles_caches:: ELementsCaches
    device_dh::DHType
end

function Thunderbolt.init_linear_operator(strategy::CudaAssemblyStrategy,protocol::IntegrandType,qrc::QuadratureRuleCollection,dh::AbstractDofHandler ) where {IntegrandType}
    if CUDA.functional()
        FT = floattype(strategy)
        b = CUDA.zeros(FT, ndofs(dh))
        linear_op =  LinearOperator(b, protocol, qrc, dh)
        return _init_linop_cuda(linear_op,strategy)
    else
        error("CUDA is not functional, please check your GPU driver and CUDA installation")
    end
end

function Thunderbolt.init_linear_operator(strategy::CudaAssemblyStrategy,linop::LinearOperator) 
    ## TODO: Dunno if this is useful or not
    if CUDA.functional()
        @unpack b, qrc, dh, integrand  = linop
        linear_op =  LinearOperator(b |> cu, protocol, qrc, dh)
        return _init_linop_cuda(linear_op,strategy)
    else
        error("CUDA is not functional, please check your GPU driver and CUDA installation")
    end
end

function _init_linop_cuda(linop::LinearOperator,strategy::CudaAssemblyStrategy)
    IT = inttype(strategy)
    FT = floattype(strategy)
    @unpack dh  = linop
    n_cells = dh |> get_grid |> getncells |> (x -> convert(IT, x))
    threads = convert(IT, min(n_cells, 256))
    blocks = _calculate_nblocks(threads, n_cells)
    n_basefuncs = convert(IT,ndofs_per_cell(dh)) 
    eles_caches = _setup_caches(linop)
    device_dh = Adapt.adapt_structure(strategy, dh)
    mem_alloc = try_allocate_shared_mem(FeMemShape{FT}, threads, n_basefuncs)
    mem_alloc isa Nothing || return CudaOperatorKernel(linop, threads, blocks, mem_alloc,eles_caches,device_dh)

    mem_alloc =allocate_global_mem(FeMemShape{FT}, blocks * threads, n_basefuncs) # allocate global memory only for the active threads
    return CudaOperatorKernel(linop, threads, blocks, mem_alloc,eles_caches,device_dh)
end


function _calculate_nblocks(threads::Ti, n_cells::Ti) where {Ti <: Integer}
    dev = device()
    no_sms = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    required_blocks = cld(n_cells, threads)
    required_blocks < 2 * no_sms || return convert(Ti, 2 * no_sms)
    return convert(Ti, required_blocks)
end


function _setup_caches(op::LinearOperator)
    @unpack b, qrc,dh,  integrand  = op
    sdh_to_cache = sdh  -> 
    begin
        # Prepare evaluation caches
        ip          = Ferrite.getfieldinterpolation(sdh, sdh.field_names[1])
        element_qr  = getquadraturerule(qrc, sdh)
        
        # Build evaluation caches
        element_cache =  Adapt.adapt_structure(CuArray,setup_element_cache(integrand, element_qr, ip, sdh))
        return element_cache
    end
    eles_caches  = dh.subdofhandlers .|> sdh_to_cache 
    return eles_caches |> cu
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

(op_ker::CudaOperatorKernel)(time) = update_operator!(op_ker.op, time)

function Thunderbolt.update_operator!(op_ker::CudaOperatorKernel{<:LinearOperator}, time)
    @unpack op, threads, blocks, mem_alloc,eles_caches,device_dh = op_ker
    @unpack b  = op
    ker = () -> _update_linear_operator_kernel!(b, device_dh, eles_caches,mem_alloc, time)
    _launch_kernel!(ker, threads, blocks, mem_alloc)
end


function _update_linear_operator_kernel!(b, dh_, eles_caches,mem_alloc, time)
    dh = dh_.gpudata
    for sdh_idx in 1:length(dh.subdofhandlers)
        element_cache = eles_caches[sdh_idx]
        for cell in CellIterator(dh, convert(Int32,sdh_idx), mem_alloc)
            bₑ = cellfe(cell)
            assemble_element!(bₑ, cell, element_cache, time)
            dofs = celldofs(cell)
            @inbounds for i in 1:length(dofs)
                b[dofs[i]] += bₑ[i]
            end
        end
    end
    return nothing
end
