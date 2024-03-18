@testset "One Max" begin 

   f(x::BitVector) = sum(x)

   ga = BinGA.BinaryGeneticAlgorithm(100, 100, f)

   BinGA.step!(ga)
   
   ave1 = BinGA.averagecost(ga)

   BinGA.step!(ga)

   ave2 = BinGA.averagecost(ga)

   BinGA.step!(ga)

   ave3 = BinGA.averagecost(ga)

   @info ave1 ave2 ave3
   @test ave3 <= ave2 <= ave1
end 