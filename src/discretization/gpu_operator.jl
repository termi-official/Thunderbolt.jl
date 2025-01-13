# TODO: move to extension
# TODO: Type stability

###################################
# GPU dispatch for LinearOperator #
###################################

# interfaces #
abstract type AbstractOperatorKernel{BKD} end

function init_linear_operator(::Type{BKD},protocol::IntegrandType,qrc::QuadratureRuleCollection,dh::AbstractDofHandler ) where {BKD,IntegrandType}
    error("Not implemented")
end

function init_linear_operator(::Type{BKD}, linop::LinearOperator) where {BKD}
    error("Not implemented")
end

function update_operator!(::AbstractOperatorKernel, time)
    error("Not implemented")
end

# CUDA backend concrete implementation #
struct CudaOperatorKernel{Operator, Ti <: Integer, MemAlloc,ELementsCaches} <: AbstractOperatorKernel{CUDABackend} 
    op::Operator
    threads::Ti
    blocks::Ti
    mem_alloc::MemAlloc
    eles_caches:: ELementsCaches
end

function init_linear_operator(::Type{CUDABackend},protocol::IntegrandType,qrc::QuadratureRuleCollection,dh::AbstractDofHandler ) where {IntegrandType}
    if CUDA.functional()
        b = CUDA.zeros(Float32, ndofs(dh))
        linear_op =  LinearOperator(b, protocol, qrc, dh)
        return _init_linop_cuda(linear_op)
    else
        error("CUDA is not functional, please check your GPU driver and CUDA installation")
    end
end

function init_linear_operator(::Type{CUDABackend},linop::LinearOperator) 
    ## TODO: Dunno if this is useful or not
    if CUDA.functional()
        @unpack b, qrc, dh, integrand  = linop
        linear_op =  LinearOperator(b |> cu, protocol, qrc, dh)
        return _init_linop_cuda(linear_op)
    else
        error("CUDA is not functional, please check your GPU driver and CUDA installation")
    end
end

function _init_linop_cuda(linop::LinearOperator)
    @unpack dh  = linop
    n_cells = dh |> get_grid |> getncells |> Int32
    threads = convert(Int32, min(n_cells, 256))
    blocks = _calculate_nblocks(threads, n_cells)
    n_basefuncs = ndofs_per_cell(dh) |> Int32
    eles_caches = _setup_caches(linop)
    mem_alloc = FerriteUtils.try_allocate_shared_mem(FerriteUtils.RHSObject{Float32}, threads, n_basefuncs)
    mem_alloc isa Nothing || return CudaOperatorKernel(linop, threads, blocks, mem_alloc,eles_caches)

    mem_alloc = FeriteUtils.allocate_global_mem(FerriteUtils.RHSObject{Float32}, n_cells, n_basefuncs)
    return CudaOperatorKernel(linop, threads, blocks, mem_alloc,eles_caches)
end


function _calculate_nblocks(threads::Ti, n_cells::Ti) where {Ti <: Integer}
    dev = device()
    no_sms = CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    required_blocks = cld(n_cells, threads)
    required_blocks < 2 * no_sms || return convert(Ti, 2 * no_sms)
    return convert(Ti, required_blocks)
end


function _setup_caches(op::LinearOperator)
    @unpack b, qrc, dh, integrand  = op
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

(op_ker::CudaOperatorKernel)(time) = update_operator!(op_ker.op, time)

function update_operator!(op_ker::CudaOperatorKernel, time)
    @unpack op, threads, blocks, mem_alloc,eles_caches = op_ker
    @unpack b, qrc, dh, integrand  = op
    ## TODO: element_caches should be in the init or Lazily evaluated here and then stored in CUDA operator object
    #eles_caches =Adapt.adapt_structure(CuArray,_setup_caches(op) |> cu)
    ker = () -> _update_linear_operator_kernel!(b, dh, eles_caches,mem_alloc, time)
    _launch_kernel!(ker, threads, blocks, mem_alloc)
end



function _update_linear_operator_kernel!(b, dh, eles_caches,mem_alloc, time)
    for sdh_idx in 1:length(dh.subdofhandlers)
        #sdh = dh.subdofhandlers[sdh_idx]
        element_cache = eles_caches[sdh_idx]
        #ndofs = ndofs_per_cell(sdh) ## TODO: check memalloc whether rhs is a constant vector or not ? 
        for cell in CellIterator(dh,convert(Int32, sdh_idx) ,mem_alloc)
            bₑ = FerriteUtils.cellfe(cell)
            assemble_element!(bₑ, cell, element_cache, time)
            #b[celldofs(cell)] .+= bₑ
            dofs = celldofs(cell)
            @inbounds for i in 1:length(dofs)
                b[dofs[i]] += bₑ[i]
            end
        end
    end
    return nothing
end


## TODO: put the adapt somewhere else ?!
function Adapt.adapt_structure(to, element_cache::AnalyticalCoefficientElementCache)
    cc = Adapt.adapt_structure(to, element_cache.cc)
    nz_intervals = Adapt.adapt_structure(to, element_cache.nonzero_intervals |> cu)
    cv = element_cache.cv
    fv = Adapt.adapt(to, FerriteUtils.StaticInterpolationValues(cv.fun_values))
    gm = Adapt.adapt(to, FerriteUtils.StaticInterpolationValues(cv.geo_mapping))
    n_quadoints = cv.qr.weights |> length
    weights = Adapt.adapt(to, ntuple(i -> cv.qr.weights[i], n_quadoints))
    sv = FerriteUtils.StaticCellValues(fv, gm, weights)
    #cv = Adapt.adapt_structure(to, element_cache.cv)
    return AnalyticalCoefficientElementCache(cc, nz_intervals, sv)
end

function Adapt.adapt_structure(to, coeff::AnalyticalCoefficientCache)
    f = Adapt.adapt_structure(to, coeff.f)
    coordinate_system_cache = Adapt.adapt_structure(to, coeff.coordinate_system_cache)
    return AnalyticalCoefficientCache(f, coordinate_system_cache)
end

function Adapt.adapt_structure(to, cysc::CartesianCoordinateSystemCache)
    cs = Adapt.adapt_structure(to, cysc.cs)
    cv = Adapt.adapt_structure(to, cysc.cv)
    return CartesianCoordinateSystemCache(cs, cv)
end

function Adapt.adapt_structure(to, cv::CellValues)
    fv = Adapt.adapt(to, FerriteUtils.StaticInterpolationValues(cv.fun_values))
    gm = Adapt.adapt(to, FerriteUtils.StaticInterpolationValues(cv.geo_mapping))
    n_quadoints = cv.qr.weights |> length
    weights = Adapt.adapt(to, ntuple(i -> cv.qr.weights[i], n_quadoints))
    return FerriteUtils.StaticCellValues(fv, gm, weights)
end

function _launch_kernel!(ker, threads, blocks, ::FerriteUtils.AbstractGlobalMemAlloc)
    CUDA.@cuda threads=threads blocks=blocks ker()
end

function _launch_kernel!(ker, threads, blocks, mem_alloc::FerriteUtils.AbstractSharedMemAlloc)
    shmem_size = FerriteUtils.mem_size(mem_alloc)
    CUDA.@sync CUDA.@cuda threads=threads blocks=blocks  shmem = shmem_size ker()
end
