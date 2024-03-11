using JSON, CSV, DataFrames

function jsons_to_csv(json_files, output_csv)
    # Placeholder DataFrame initialization
    df = DataFrame()
    first_file = true
    kernel_names = []

    for file in json_files
        if isfile(file)
            # Read the JSON file
            contents = JSON.parsefile(file)

            kernel_name = splitext(basename(file))[1]
            push!(kernel_names, kernel_name)

            # For the first file, initialize the DataFrame with column names and types
            if first_file
                for (key, value) in contents
                    df[!, Symbol(key)] = Vector{typeof(value)}()
                end
                first_file = false
            end
            
            # Create a new row for the DataFrame
            row = [contents[key] for key in keys(contents)]
            
            # Append the row to the DataFrame
            push!(df, row)
        else
            println("File does not exist: ", file)
        end
    end

    kernel_names = replace.(kernel_names, "_" => "-")
    df = insertcols!(df, 1, :KernelName => kernel_names)

    if !isdir("csv")
        mkdir("csv")
    end

    # Write the DataFrame to a CSV file
    CSV.write("csv/" * output_csv, df)
end

function generate_benchmark_csv(benchmark, kernels, suffix)
    kernel_files = []
    for kernel in kernels
        push!(kernel_files, "../../julia_" * suffix * "/" * benchmark * "/" * kernel * ".json")
    end
    jsons_to_csv(kernel_files, benchmark * "_" * suffix * ".csv")
end

function generate_benchmarks_csv(suffix)
    generate_benchmark_csv("backprop", ["bpnn_layerforward_CUDA", "bpnn_adjust_weights_cuda"], suffix)
    generate_benchmark_csv("bfs", ["Kernel_$(i)" for i in 0:11], suffix)
    generate_benchmark_csv("bfs", ["Kernel2_$(i)" for i in 0:11], suffix)
    generate_benchmark_csv("hotspot", ["calculate_temp"], suffix)
    generate_benchmark_csv("leukocyte", ["GICOV_kernel", "dilate_kernel", "IMGVF_kernel"], suffix)
    generate_benchmark_csv("lud", ["lud_diagonal", "lud_perimeter", "lud_internal"], suffix)
    generate_benchmark_csv("nn", ["euclid"], suffix)
    generate_benchmark_csv("nw", ["needle_cuda_shared_1", "needle_cuda_shared_2"], suffix)
    generate_benchmark_csv("particlefilter", ["likelihood_kernel", "sum_kernel", "normalize_weights_kernel", "find_index_kernel"], suffix)
    generate_benchmark_csv("pathfinder", ["dynproc_kernel"], suffix)
    generate_benchmark_csv("streamcluster", ["kernel_compute_cost"], suffix)
end

generate_benchmarks_csv("cuda")
generate_benchmarks_csv("gen")