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

function generate_benchmark_csv(suffix)
    backprop_files = ["../../julia_" * suffix * "/backprop/bpnn_layerforward_CUDA.json",
                      "../../julia_" * suffix * "/backprop/bpnn_adjust_weights_cuda.json"]
    jsons_to_csv(backprop_files, "backprop_" * suffix * ".csv")

    bfs_files = ["../../julia_" * suffix * "/bfs/Kernel.json",
                 "../../julia_" * suffix * "/bfs/Kernel2.json"]
    jsons_to_csv(bfs_files, "bfs_" * suffix * ".csv")

    hotspot_files = ["../../julia_" * suffix * "/hotspot/calculate_temp.json"]
    jsons_to_csv(hotspot_files, "hotspot_" * suffix * ".csv")

    leukocyte_files = ["../../julia_" * suffix * "/leukocyte/GICOV_kernel.json",
                       "../../julia_" * suffix * "/leukocyte/dilate_kernel.json",
                       "../../julia_" * suffix * "/leukocyte/IMGVF_kernel.json"]
    jsons_to_csv(leukocyte_files, "leukocyte_" * suffix * ".csv")

    lud_files = ["../../julia_" * suffix * "/lud/lud_diagonal.json",
                 "../../julia_" * suffix * "/lud/lud_perimeter.json",
                 "../../julia_" * suffix * "/lud/lud_internal.json"]
    jsons_to_csv(lud_files, "lud_" * suffix * ".csv")

    euclid_files = ["../../julia_" * suffix * "/nn/euclid.json"]
    jsons_to_csv(euclid_files, "nn_" * suffix * ".csv")

    nw_files = ["../../julia_" * suffix * "/nw/needle_cuda_shared_1.json",
                "../../julia_" * suffix * "/nw/needle_cuda_shared_2.json"]
    jsons_to_csv(nw_files, "nw_" * suffix * ".csv")

    particle_filter_files = ["../../julia_" * suffix * "/particlefilter/likelihood_kernel.json",
                             "../../julia_" * suffix * "/particlefilter/sum_kernel.json",
                             "../../julia_" * suffix * "/particlefilter/normalize_weights_kernel.json",
                             "../../julia_" * suffix * "/particlefilter/find_index_kernel.json"]
    jsons_to_csv(particle_filter_files, "particle_filter_" * suffix * ".csv")

    pathfinder_files = ["../../julia_" * suffix * "/pathfinder/dynproc_kernel.json"]
    jsons_to_csv(pathfinder_files, "pathfinder_" * suffix * ".csv")

    stream_cluster_files = ["../../julia_" * suffix * "/streamcluster/kernel_compute_cost.json"]
    jsons_to_csv(stream_cluster_files, "stream_cluster_" * suffix * ".csv")
end

generate_benchmark_csv("cuda")
generate_benchmark_csv("gen")