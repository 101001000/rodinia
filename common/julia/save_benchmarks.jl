using JSON, CSV, DataFrames

global_cuda_df = DataFrame()
global_gen_df = DataFrame()
global_cuda_kernel_names = []
global_gen_kernel_names = []

function jsons_to_csv(json_files, output_csv, suffix)
    # Placeholder DataFrame initialization
    df = DataFrame()
    first_file = true
    kernel_names = []

    for file in json_files
        if isfile(file)
            # Read the JSON file
            contents = JSON.parsefile(file)


            if suffix == "cuda"
                if isempty(global_cuda_kernel_names)
                    for (key, value) in contents
                        global_cuda_df[!, Symbol(key)] = Vector{typeof(value)}()
                    end
                end
            else
                if isempty(global_gen_kernel_names)
                    for (key, value) in contents
                        global_gen_df[!, Symbol(key)] = Vector{typeof(value)}()
                    end
                end
            end

            

            kernel_name = splitext(basename(file))[1]
            push!(kernel_names, kernel_name)

            if suffix == "cuda"
                push!(global_cuda_kernel_names, kernel_name)
            else
                push!(global_gen_kernel_names, kernel_name)
            end
            

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
            
            if suffix == "cuda"
                push!(global_cuda_df, row)
            else
                push!(global_gen_df, row)
            end
            
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
    jsons_to_csv(kernel_files, benchmark * "_" * suffix * ".csv", suffix)
end

function generate_benchmarks_csv(suffix)
    generate_benchmark_csv("backprop", ["bpnn_layerforward_CUDA", "bpnn_adjust_weights_cuda"], suffix)
    generate_benchmark_csv("bfs", ["Kernel"], suffix)
    generate_benchmark_csv("bfs", ["Kernel2"], suffix)
    generate_benchmark_csv("hotspot", ["calculate_temp"], suffix)
    generate_benchmark_csv("leukocyte", ["GICOV_kernel", "dilate_kernel", "IMGVF_kernel"], suffix)
    generate_benchmark_csv("lud", ["lud_diagonal", "lud_perimeter", "lud_internal"], suffix)
    generate_benchmark_csv("nn", ["euclid"], suffix)
    generate_benchmark_csv("nw", ["needle_cuda_shared_1", "needle_cuda_shared_2"], suffix)
    generate_benchmark_csv("particlefilter", ["likelihood_kernel", "sum_kernel", "normalize_weights_kernel", "find_index_kernel"], suffix)
    generate_benchmark_csv("pathfinder", ["dynproc_kernel"], suffix)
    generate_benchmark_csv("streamcluster", ["kernel_compute_cost"], suffix)
end

# Iterate through all the rows in both dataframes finding the matching one. 
# Normalize the average value along the first one.
function normalize_dfs!(df1, df2)

    insertcols!(df1, ncol(df1) + 1, :overhead => 0.0)
    insertcols!(df2, ncol(df2) + 1, :overhead => 0.0)

    for row1 in eachrow(df1)

        processed_row = false

        for row2 in eachrow(df2)
            if row1["KernelName"] == row2["KernelName"]
                nf = row1["median"] / 100
                row1["median"] /= nf 
                row2["median"] /= nf
                row1["overhead"] = row1["median"] - row2["median"]
                row2["overhead"] = row2["median"] - row1["median"]
                row1["std"] /= nf
                row2["std"] /= nf
                processed_row = true
            end
        end

        if !processed_row
            nf = row1["median"] / 100
            row1["median"] /= nf
            row1["std"] /= nf
            println("Row without match, " * string(row1["KernelName"]))
        end
    end
end

generate_benchmarks_csv("cuda")
generate_benchmarks_csv("gen")


global_cuda_kernel_names = replace.(global_cuda_kernel_names, "_" => "-")
global_gen_kernel_names = replace.(global_gen_kernel_names, "_" => "-")

global_cuda_df = insertcols!(global_cuda_df, 1, :KernelName => global_cuda_kernel_names)
global_gen_df = insertcols!(global_gen_df, 1, :KernelName => global_gen_kernel_names)

normalize_dfs!(global_cuda_df, global_gen_df)


CSV.write("csv/aggregated-set-cuda.csv", global_cuda_df)
CSV.write("csv/aggregated-set-gen.csv", global_gen_df)