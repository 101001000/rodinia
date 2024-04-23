#!/usr/bin/env julia

using CUDA, NVTX
using LLVM

include("common.jl")
include("lud_kernel.jl")
include("../../common/julia/utils.jl")

function main(args)
    println("WG size of kernel = $BLOCK_SIZE X $BLOCK_SIZE")

    verify = haskey(ENV, "OUTPUT")

    if length(args) == 1
        try
            matrix_dim, input_file = parse(Int, args[1]), nothing
        catch
            matrix_dim, input_file = nothing, args[1]
        end
    else
        matrix_dim, input_file = 32, nothing
    end

    if input_file != nothing
        println("Reading matrix from file $input_file")
        matrix, matrix_dim = create_matrix_from_file(input_file)
    elseif matrix_dim > 0
        println("Creating matrix internally size=$(matrix_dim)")
        matrix = create_matrix(matrix_dim)
    else
        error("No input file specified!")
    end

    if verify
        println("Before LUD")
        matrix_copy = copy(matrix)
    end

    sec = CUDA.@elapsed begin
        d_matrix = CuArray(matrix)
        lud_cuda(d_matrix, matrix_dim)
        matrix = Array(d_matrix)
    end
    println("Time consumed(ms): $(1000sec)")

    println("lud_diagonal")
    display(aggregate_benchmarks(lud_diagonal_benchmarks))
    save_benchmark(aggregate_benchmarks(lud_diagonal_benchmarks), "lud_diagonal.json")
    println("lud_perimeter")
    display(aggregate_benchmarks(lud_perimeter_benchmarks))
    save_benchmark(aggregate_benchmarks(lud_perimeter_benchmarks), "lud_perimeter.json")
    println("lud_internal")
    display(aggregate_benchmarks(lud_internal_benchmarks))
    save_benchmark(aggregate_benchmarks(lud_internal_benchmarks), "lud_internal.json")

    save_benchmark(aggregate_benchmarks([lud_diagonal_benchmarks; lud_perimeter_benchmarks; lud_internal_benchmarks]), "lud-aggregated.json")

    if verify
        println("After LUD")
        println(">>>Verify<<<<")
        lud_verify(matrix_copy, matrix, matrix_dim)
    end
end

# FIXME: for now we increase the unroll threshold to ensure that the nested loops in the
# kernels are unrolled as is the case for the CUDA benchmark. Ideally, we should annotate
# the loops or the kernel(s) with the @unroll macro once it is available.
LLVM.clopts("--unroll-threshold=1200")

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
