struct InternalVariableInfo
    name::Symbol
    size::Int
end

function _compatible_cellset(dh::DofHandler, firstcell::Int)
    for sdh in dh.subdofhandlers
        if firstcell ∈ sdh.cellset
            return sdh.cellset
        end
    end
    error("Cell $firstcell not found.")
end
