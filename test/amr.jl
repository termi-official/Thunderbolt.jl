using Thunderbolt
using Test

function test_amr_line()
    mesh = ForestMesh(generate_grid(Line, (3,)))
    elements_initial = ForestElementIndex[]
    for fi in Thunderbolt.ForestIterator(mesh)
        push!(elements_initial, deepcopy(fi.current_element_idx))
    end
    @test elements_initial == [ForestElementIndex(1, []),
                               ForestElementIndex(2, []),
                               ForestElementIndex(3, [])]

    refine_isotropic!(mesh, ForestElementIndex(1, []))
    elements_ref1 = ForestElementIndex[]
    for fi in Thunderbolt.ForestIterator(mesh)
        push!(elements_ref1, deepcopy(fi.current_element_idx))
    end
    @test elements_ref1 == [ForestElementIndex(1, []),
                              ForestElementIndex(1, [1]),
                              ForestElementIndex(1, [2]),
                            ForestElementIndex(2, []),
                            ForestElementIndex(3, [])]

    refine_isotropic!(mesh, ForestElementIndex(1, [2]))
    elements_ref2 = ForestElementIndex[]
    for fi in Thunderbolt.ForestIterator(mesh)
        push!(elements_ref2, deepcopy(fi.current_element_idx))
    end
    @test elements_ref2 == [ForestElementIndex(1, []),
                              ForestElementIndex(1, [1]),
                              ForestElementIndex(1, [2]),
                                ForestElementIndex(1, [2, 1]),
                                ForestElementIndex(1, [2, 2]),
                            ForestElementIndex(2, []),
                            ForestElementIndex(3, [])]

    refine_isotropic!(mesh, ForestElementIndex(1, [2, 1]))
    elements_ref3 = ForestElementIndex[]
    for fi in Thunderbolt.ForestIterator(mesh)
        push!(elements_ref3, deepcopy(fi.current_element_idx))
    end
    @test elements_ref3 == [ForestElementIndex(1, []),
                              ForestElementIndex(1, [1]),
                              ForestElementIndex(1, [2]),
                                ForestElementIndex(1, [2, 1]),
                                  ForestElementIndex(1, [2, 1, 1]),
                                  ForestElementIndex(1, [2, 1, 2]),
                                ForestElementIndex(1, [2, 2]),
                            ForestElementIndex(2, []),
                            ForestElementIndex(3, [])]

    derefine!(mesh, ForestElementIndex(1, [2, 1]))
    elements_deref = ForestElementIndex[]
    for fi in Thunderbolt.ForestIterator(mesh)
        push!(elements_deref, deepcopy(fi.current_element_idx))
    end
    @test elements_deref == [ForestElementIndex(1, []),
                              ForestElementIndex(1, [1]),
                              ForestElementIndex(1, [2]),
                                ForestElementIndex(1, [2, 1]),
                                ForestElementIndex(1, [2, 2]),
                            ForestElementIndex(2, []),
                            ForestElementIndex(3, [])]
end
test_amr_line()
