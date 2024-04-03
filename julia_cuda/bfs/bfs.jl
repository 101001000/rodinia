#!/usr/bin/env julia

using CUDA, NVTX
using BenchmarkTools

include("../../common/julia/utils.jl")

const OUTPUT = haskey(ENV, "OUTPUT")

# configuration
const MAX_THREADS_PER_BLOCK = UInt32(256)

Kernel_benchmarks = []
Kernel2_benchmarks = []

struct Node
    starting::Int32
    no_of_edges::Int32
end

function Kernel(g_graph_nodes, g_graph_edges, g_graph_mask_i, g_graph_mask_o,
    g_updating_graph_mask, g_graph_visited, g_cost_i, g_cost_o,
    no_of_nodes)
    tid = (blockIdx().x - 1) * MAX_THREADS_PER_BLOCK + threadIdx().x
    if tid <= no_of_nodes && g_graph_mask_i[tid]
        g_graph_mask_o[tid] = false
        for i = g_graph_nodes[tid].starting:(g_graph_nodes[tid].starting+g_graph_nodes[tid].no_of_edges-Int32(1))
            id = g_graph_edges[i]
            if !g_graph_visited[id]
                g_cost_o[id] = g_cost_i[tid] + Int32(1)
                g_updating_graph_mask[id] = true
            end
        end
    end
	return
end

function Kernel2(g_graph_mask, g_updating_graph_mask_i, g_updating_graph_mask_o, g_graph_visited,
    g_over, no_of_nodes)
    tid = (blockIdx().x - UInt32(1)) * MAX_THREADS_PER_BLOCK + threadIdx().x
    if tid <= no_of_nodes && g_updating_graph_mask_i[tid]
        g_graph_mask[tid] = true
        g_graph_visited[tid] = true
        g_over[1] = true
        g_updating_graph_mask_o[tid] = false
    end
    return
end

function read_file(input_f)
    fp = open(input_f)

    no_of_nodes = parse(Int, readline(fp))

    # allocate host memory
    h_graph_nodes = Vector{Node}(undef, no_of_nodes)
    h_graph_mask = Vector{Bool}(undef, no_of_nodes)
    h_updating_graph_mask = Vector{Bool}(undef, no_of_nodes)
    h_graph_visited = Vector{Bool}(undef, no_of_nodes)
    h_cost = Vector{Int32}(undef, no_of_nodes)

    # initalize the memory
    for i = 1:no_of_nodes
        start = parse(Int, readuntil(fp, ' '))
        edgeno = parse(Int, readline(fp))
        h_graph_nodes[i] = Node(start + 1, edgeno)
        h_graph_mask[i] = false
        h_updating_graph_mask[i] = false
        h_graph_visited[i] = false
        h_cost[i] = -1
    end

    skipchars(isspace, fp)

    # read the source node from the file
    source = parse(Int, readline(fp))
    source += 1

    skipchars(isspace, fp)

    # set the source node as true in the mask
    h_graph_mask[source] = true
    h_graph_visited[source] = true
    h_cost[source] = 0

    edge_list_size = parse(Int, readline(fp))

    skipchars(isspace, fp)

    h_graph_edges = Vector{Int32}(undef, edge_list_size)
    for i = 1:edge_list_size
        id = parse(Int, readuntil(fp, ' '))
        cost = parse(Int, readline(fp))
        h_graph_edges[i] = id + 1
    end

    close(fp)

    return no_of_nodes, h_graph_nodes, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost, h_graph_edges
end

function main(args)
    if length(args) != 1
        error("Usage: bfs.jl <input_file>")
    end
    input_f = args[1]

    println("Reading File")
    no_of_nodes, h_graph_nodes, h_graph_mask, h_updating_graph_mask, h_graph_visited, h_cost, h_graph_edges = read_file(input_f)
    println("Read File")

    num_of_blocks = 1
    num_of_threads_per_block = no_of_nodes

    # Make execution Parameters according to the number of nodes
    # Distribute threads across multiple Blocks if necessary
    if no_of_nodes > MAX_THREADS_PER_BLOCK
        num_of_blocks = ceil(Integer, no_of_nodes / MAX_THREADS_PER_BLOCK)
        num_of_threads_per_block = MAX_THREADS_PER_BLOCK
    end

    # setup execution parameters
    blocks = num_of_blocks
    threads = num_of_threads_per_block


    # Manual copies to device
    g_graph_nodes = CuArray(h_graph_nodes)
    g_graph_edges = CuArray(h_graph_edges)
    g_graph_mask = CuArray(h_graph_mask)
    g_updating_graph_mask = CuArray(h_updating_graph_mask)
    g_graph_visited = CuArray(h_graph_visited)
    g_cost = CuArray(h_cost)
    g_stop = CuArray{Bool}(undef, 1)


    k = 0
    println("Start traversing the tree")
    stop = Bool[1]

	g_graph_mask_o = CUDA.zeros(Bool, size(g_graph_mask))
	g_cost_o = CUDA.zeros(Int32, size(g_cost))
	g_updating_graph_mask_o = CUDA.zeros(Bool, size(g_updating_graph_mask))

    while true

        print("Progress estimated: $(100*k/12)%   \r")
        flush(stdout)

        # if no thread changes this value then the loop stops
        stop[1] = false
        copyto!(g_stop, stop)

        b = @benchmark (CUDA.@sync @cuda blocks = $blocks threads = $threads Kernel(
            $g_graph_nodes, $g_graph_edges, $g_graph_mask, $g_graph_mask_o,
            $g_updating_graph_mask, $g_graph_visited,
            $g_cost, $g_cost_o, $no_of_nodes
        )) samples=10000
        push!(Kernel_benchmarks, b)
		                
		CUDA.copy!(g_graph_mask, g_graph_mask_o)
		CUDA.copy!(g_cost, g_cost_o)

        b = @benchmark (CUDA.@sync @cuda blocks = $blocks threads = $threads Kernel2(
            $g_graph_mask, $g_updating_graph_mask, $g_updating_graph_mask_o, $g_graph_visited,
            $g_stop, $no_of_nodes
        )) samples=10000

        push!(Kernel2_benchmarks, b)
		
		CUDA.copy!(g_updating_graph_mask, g_updating_graph_mask_o)

        k += 1
        copyto!(stop, g_stop)
        if !stop[1]
            break
        end
    end

    # Copy result back + free
    h_cost = Array(g_cost)

    println("Kernel Executed $k times")

    println("Kernel")
    display(aggregate_benchmarks(Kernel_benchmarks))
    save_benchmark(aggregate_benchmarks(Kernel_benchmarks), "Kernel.json")
    println("Kernel2")
    display(aggregate_benchmarks(Kernel2_benchmarks))
    save_benchmark(aggregate_benchmarks(Kernel2_benchmarks), "Kernel2.json")

    save_benchmarks_accum([Kernel_benchmarks; Kernel2_benchmarks], "bfs-aggregated.json")

    # Store the result into a file
    if OUTPUT
        open("output.txt", "w") do fpo
            for i = 1:no_of_nodes
                write(fpo, "$(i-1)) cost:$(h_cost[i])\n")
            end
        end
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
