using BenchmarkTools
using JSON
using Statistics
using Random

function aggregate_benchmarks(benchmarks)
    combined_times = []
    combined_gctimes = []
    combined_memory = 0
    combined_allocs = 0
    min_samples = length(first(benchmarks).times)
    for b in benchmarks
        if min_samples != length(b.times)
            println("Warning, inconsistent benchmark sampling, droping random samples. ")
        end
        min_samples = min(min_samples, length(b.times))
    end
    for b in benchmarks
        times_c = copy(b.times)
        shuffle!(times_c)
        append!(combined_times, times_c[1:min_samples])
        append!(combined_gctimes, b.gctimes)
        combined_allocs += b.allocs
        combined_memory += b.memory
    end
    return BenchmarkTools.Trial(BenchmarkTools.Parameters(; samples=length(combined_times), evals=1), combined_times, combined_gctimes, combined_memory, combined_allocs)
end

function save_benchmark(benchmark, filename)
    b_dict = Dict("max" => maximum(benchmark).time / 1000,
     "min" => minimum(benchmark).time / 1000,
     "mean" => Statistics.mean(benchmark).time / 1000,
     "var" => Statistics.var(benchmark).time / 1000,
     "median" => Statistics.median(benchmark).time / 1000,
     "std" => Statistics.std(benchmark).time / 1000,
     "samples" => length(benchmark.times),
     "memory" => benchmark.memory,
     "allocs" => benchmark.allocs)

    json_string = JSON.json(b_dict)

    if isfile(filename)
        println("Warning, overriding existing results")
    end
    
    open(filename,"w") do f 
        write(f, json_string) 
    end
end
