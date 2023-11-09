import Thunderbolt: AssembledNonlinearOperator, AssembledBilinearOperator, NullOperator, DiagonalOperator, BlockOperator
import LinearAlgebra: mul!
using BlockArrays, SparseArrays

@testset "Operators" begin
    @testset "Actions" begin
        vin = ones(5)
        vout = ones(5)

        nullop = NullOperator{Float64,5,5}()
        @test eltype(nullop) == Float64
        @test length(vin)  == size(nullop, 1)
        @test length(vout) == size(nullop, 2)

        mul!(vout, nullop, vin)
        @test vout == zeros(5)

        vout .= ones(5)
        mul!(vout, nullop, vin, 2.0, 1.0)
        @test vout == ones(5)

        @test length(vin)  == size(nullop, 1)
        @test length(vout) == size(nullop, 2)
        
        @test Thunderbolt.getJ(nullop) ≈ zeros(5,5)


        diagop = DiagonalOperator([1.0, 2.0, 3.0, 4.0, 5.0])
        @test length(vin)  == size(diagop, 1)
        @test length(vout) == size(diagop, 2)
        mul!(vout, diagop, vin)
        @test vout == [1.0, 2.0, 3.0, 4.0, 5.0]

        mul!(vout, diagop, vin, 1.0, 1.0)
        @test vout == 2.0 .* [1.0, 2.0, 3.0, 4.0, 5.0]

        mul!(vout, diagop, vin, -2.0, 1.0)
        @test vout == zeros(5)
        @test length(vin)  == size(diagop, 1)
        @test length(vout) == size(diagop, 2)
        
        @test Thunderbolt.getJ(diagop) ≈ spdiagm([1.0, 2.0, 3.0, 4.0, 5.0])


        vin = ones(4)
        vout .= ones(5)
        nullop_rect = NullOperator{Float64,4,5}()

        @test length(vin)  == size(nullop_rect, 1)
        @test length(vout) == size(nullop_rect, 2)
        @test vout == vout
        @test length(vin)  == size(nullop_rect, 1)
        @test length(vout) == size(nullop_rect, 2)

        @test Thunderbolt.getJ(nullop_rect) ≈ zeros(4,5)


        vin = mortar([ones(4), -ones(2)])
        vout = mortar([-ones(4), -ones(2)])
        bop_id = BlockOperator((
            DiagonalOperator(ones(4)), NullOperator{Float64,2,4}(),
            NullOperator{Float64,4,2}(),  DiagonalOperator(ones(2))
        ))

        @test vout != vin
        mul!(vout, bop_id, vin)
        @test vout == vin
        mul!(vout, bop_id, vin, 2.0, 1.0)
        @test vout == 3.0*vin
        
        @test Thunderbolt.getJ(bop_id) ≈ spdiagm(ones(6))
    end
end
