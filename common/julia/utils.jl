

using BenchmarkTools

function aggregate_benchmarks(benchmarks)

    combined_times = []
    combined_gctimes = []
    combined_memory = 0
    combined_allocs = 0

    for b in benchmarks
        append!(combined_times, b.times)
        append!(combined_gctimes, b.gctimes)
        combined_allocs += b.allocs
        combined_memory += b.memory
    end

    return BenchmarkTools.Trial(BenchmarkTools.Parameters(; samples=length(combined_times), evals=1), combined_times, combined_gctimes, combined_memory, combined_allocs)

end

