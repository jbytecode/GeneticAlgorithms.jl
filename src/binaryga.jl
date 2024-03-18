module BinaryGA

mutable struct BinaryChromosome
    genes::BitVector
    cost::Float64
end

mutable struct BinaryGeneticAlgorithm
    population::Vector{BinaryChromosome}
    cost::Function
    selection::Function
    crossover::Function
    mutation::Function
    elitismrate::Float64
    crossoverrate::Float64
    mutationrate::Float64
end

function BinaryGeneticAlgorithm(popsize::Int, chsize::Int, costfunction::FunctionType) where {FunctionType<:Function}
    population = [BinaryChromosome(BitVector(rand(Bool, chsize))) for i in 1:popsize]
    selectionf = maketournamentselectionfunction(2)
    crossrate = 0.9
    mutaterate = 0.1
    elitisimrate = 0.1
    crossf = makeuniformcrossoverfunction(crossrate)
    mutationf = makerandommutationfunction(mutaterate)
    return BinaryGeneticAlgorithm(
        population, costfunction, selectionf, 
        crossf, mutationf, elitisimrate, crossrate, mutaterate)
end

function BinaryChromosome(genes::BitVector)
    BinaryChromosome(genes, Inf64)
end

function setcost(c::BinaryChromosome, v::Float64)
    c.cost = v
end

function setcost(c::BinaryChromosome, f::FunctionType) where {FunctionType<:Function}
    c.cost = f(c.genes)
end

function onepointcrossover(c1::BinaryChromosome, c2::BinaryChromosome, prob::Float64)::Tuple{BinaryChromosome,BinaryChromosome}
    if rand() < prob
        n = length(c1.genes)
        k = rand(1:n)
        g1 = vcat(c1.genes[1:k], c2.genes[(k+1):end])
        g2 = vcat(c2.genes[1:k], c1.genes[(k+1):end])
        return BinaryChromosome(g1), BinaryChromosome(g2)
    else
        return c1, c2
    end
end

function twopointcrossover(c1::BinaryChromosome, c2::BinaryChromosome, prob::Float64)::Tuple{BinaryChromosome,BinaryChromosome}
    if rand() < prob
        n = length(c1.genes)
        k1 = rand(1:n)
        k2 = rand(1:n)
        k1, k2 = sort([k1, k2])
        g1 = vcat(c1.genes[1:k1], c2.genes[(k1+1):k2], c1.genes[(k2+1):end])
        g2 = vcat(c2.genes[1:k1], c1.genes[(k1+1):k2], c2.genes[(k2+1):end])
        return BinaryChromosome(g1), BinaryChromosome(g2)
    else
        return c1, c2
    end
end

function uniformcrossover(c1::BinaryChromosome, c2::BinaryChromosome, prob::Float64)::Tuple{BinaryChromosome,BinaryChromosome}
    if rand() < prob
        n = length(c1.genes)
        g1 = BitVector([rand() < 0.5 ? c1.genes[i] : c2.genes[i] for i in 1:n])
        g2 = BitVector([rand() < 0.5 ? c1.genes[i] : c2.genes[i] for i in 1:n])
        return BinaryChromosome(g1), BinaryChromosome(g2)
    else
        return c1, c2
    end
end

function randommutation(c::BinaryChromosome, prob::Float64)::BinaryChromosome
    n = length(c.genes)
    g = BitVector([rand() < prob ? !c.genes[i] : c.genes[i] for i in 1:n])
    return BinaryChromosome(g)
end

function tournamentselection(pop::Vector{BinaryChromosome}, k::Int)::BinaryChromosome
    best = rand(pop)
    for i in 2:k
        c = rand(pop)
        if c.cost < best.cost
            best = c
        end
    end
    return best
end

function makeonepointcrossoverfunction(prob::Float64)::Function
    (c1::BinaryChromosome, c2::BinaryChromosome) -> onepointcrossover(c1, c2, prob)
end

function maketwopointcrossoverfunction(prob::Float64)::Function
    (c1::BinaryChromosome, c2::BinaryChromosome) -> twopointcrossover(c1, c2, prob)
end

function makeuniformcrossoverfunction(prob::Float64)::Function
    (c1::BinaryChromosome, c2::BinaryChromosome) -> uniformcrossover(c1, c2, prob)
end

function maketournamentselectionfunction(k::Int)::Function
    (pop::Vector{BinaryChromosome}) -> tournamentselection(pop, k)
end

function makerandommutationfunction(prob::Float64)::Function
    (c::BinaryChromosome) -> randommutation(c, prob)
end

function averagecost(ga::BinaryGeneticAlgorithm)::Float64
    n = length(ga.population)
    return sum(c.cost for c in ga.population) / n
end

function bestchromosome(ga::BinaryGeneticAlgorithm)::BinaryChromosome
    return argmin(ga.population, by=c -> c.cost)
end

function sortpopulation!(ga::BinaryGeneticAlgorithm)
    sort!(ga.population, by=c -> c.cost)
end


function step!(ga::BinaryGeneticAlgorithm)
    for c in ga.population
        setcost(c, ga.cost)
    end
    sortpopulation!(ga)
    n = length(ga.population)
    m = Int(ceil(ga.elitismrate * n))
    newpop = ga.population[1:m]
    while length(newpop) < n
        c1 = ga.selection(ga.population)
        c2 = ga.selection(ga.population)
        c1, c2 = ga.crossover(c1, c2)
        c1 = ga.mutation(c1)
        c2 = ga.mutation(c2)
        setcost(c1, ga.cost)
        setcost(c2, ga.cost)
        push!(newpop, c1)
        if length(newpop) < n
            push!(newpop, c2)
        end
    end
    ga.population = newpop
end 

end # End of module BinaryGA