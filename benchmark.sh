# INPUTS
#video_path="../videos/house720.mp4 -p"
#trial_name="house720-h1-3iter-data"
video_path="../videos/door1080.mov"
trial_name="door1080-h1-3iter-stream"
frames=924  # 84 frames for the 720 video, 925 for the 1080. But 1 less because it's used for the background

threads=($(seq 1 32))
iterations=3
modes=($(seq 0 6))
mode_names=("threads" "threads_pin" "ff" "ff_acc" "ff_deman" "ff_pipe_f" "omp")

output_file="136_comparison.csv"

# EXECUTE
cd ./build

# Check if outputfile exists, otherwise create it and write the header
if [ ! -f $output_file ]; then
    echo "trial name,items,id,type,threads,completion time,service time,bandwidth,speedup,scalability,efficiency" >>$output_file
fi

# Benchmark
for i in $(seq 1 $iterations); do
    echo "Iteration: $i"

    # Sequential time
    sequential_time=$(./motion-detection -s $video_path | grep -oP '(?<=computed in )\d+')
    service_time=$(echo "scale=2 ; $sequential_time / $frames" | bc)
    echo "Sequential time: $sequential_time"
    echo "$trial_name,$frames,$i,seq,1,$sequential_time,$service_time,0,1,1,1" >>$output_file

    # Parallel time with a single thread
    parallel_time_1_thread=0

    for mode in "${modes[@]}"; do
        for thread in "${threads[@]}"; do
            parallel_time=$(./motion-detection -s $video_path -w $thread -m $mode | grep -oP '(?<=computed in )\d+')
            # Update time for the paralllel time with one thread (used for scalability)
            if [ $thread -eq 1 ]; then
                parallel_time_1_thread=$parallel_time
            fi

            # Compute metrics
            service_time=$(echo "scale=2 ; $parallel_time / $frames" | bc)
            bandwidth=$(echo "scale=2 ; 1 / $service_time" | bc)
            speedup=$(echo "scale=2 ; $sequential_time / $parallel_time" | bc)
            scalability=$(echo "scale=2 ; $parallel_time_1_thread / $parallel_time" | bc)
            efficiency=$(echo "scale=2 ; $sequential_time / $thread / $parallel_time" | bc)

            # Print parallel time with thread and speedup
            echo "${mode_names[$mode]} time with $thread threads: $parallel_time, speedup: $speedup"

            # Append results to the results csv file
            echo "$trial_name,$frames,$i,${mode_names[$mode]},$thread,$parallel_time,$service_time,$bandwidth,$speedup,$scalability,$efficiency" >>$output_file
        done
    done

    ## Empty line separator
    echo "" >>$output_file
    echo ""
done