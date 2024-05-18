# This file contains the main infrastucture for coupled problems

struct EmptyTyingCache end

get_tying_dofs(::EmptyTyingCache, u) = nothing

assemble_tying!(Jₑ, residualₑ, uₑ, uₜ, cell,::EmptyTyingCache, t) = nothing
assemble_tying!(Jₑ, uₑ, uₜ, cell,::EmptyTyingCache, t) = nothing

function setup_tying_cache(tying_models::Union{<:AbstractVector,<:Tuple}, qr, ip, ip_geo)
    length(tying_models) == 0 && return EmptyTyingCache()
    return ntuple(i->setup_tying_cache(tying_models[i], qr, ip, ip_geo), length(tying_models))
end



struct RSAFDQSingleChamberTying{CVM}
    pressure_dof_index::Int
    faces::Set{FaceIndex}
    volume_method::CVM
end

struct RSAFDQTyingCache{FV <: FaceValues, CVM}
    fv::FV
    chambers::Vector{RSAFDQSingleChamberTying{CVM}}
end

struct RSAFDQTyingProblem{CVM}
    chambers::Vector{RSAFDQSingleChamberTying{CVM}}
end

solution_size(problem::RSAFDQTyingProblem) = length(problem.chambers)

function setup_tying_cache(tying_model::RSAFDQTyingProblem, qr, ip, ip_geo)
    RSAFDQTyingCache(FaceValues(qr, ip, ip_geo), tying_model.chambers)
end

function get_tying_dofs(tying_cache::RSAFDQTyingCache, u)
    return [u[chamber.pressure_dof_index] for chamber in tying_cache.chambers]
end

function assemble_tying_face_rsadfq!(Jₑ, residualₑ, uₑ, p, cell, local_face_index, fv, time)
    reinit!(fv, cell, local_face_index)

    for qp in QuadratureIterator(fv)
        assemble_face_pressure_qp!(Jₑ, uₑ, p, qp, fv)
    end
end

function assemble_tying_face_rsadfq!(Jₑ, uₑ, p, cell, local_face_index, fv, time)
    reinit!(fv, cell, local_face_index)

    for qp in QuadratureIterator(fv)
        assemble_face_pressure_qp!(Jₑ, uₑ, p, qp, fv)
    end
end

function assemble_tying!(Jₑ, residualₑ, uₑ, uₜ, cell, tying_cache::RSAFDQTyingCache, time)
    for local_face_index ∈ 1:nfaces(cell)
        for (chamber_index,chamber) in pairs(tying_cache.chambers)
            if (cellid(cell), local_face_index) ∈ chamber.faces
                assemble_tying_face_rsadfq!(Jₑ, residualₑ, uₑ, uₜ[chamber_index], cell, local_face_index, tying_cache.fv, time)
            end
        end
    end
end


function assemble_tying!(Jₑ, uₑ, uₜ, cell, tying_cache::RSAFDQTyingCache, time)
    for local_face_index ∈ 1:nfaces(cell)
        for (chamber_index,chamber) in pairs(tying_cache.chambers)
            if (cellid(cell), local_face_index) ∈ chamber.faces
                assemble_tying_face_rsadfq!(Jₑ, uₑ, uₜ[chamber_index], cell, local_face_index, tying_cache.fv, time)
            end
        end
    end
end
