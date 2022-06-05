#include <iostream>
#include <functional>

#include "opencv2/opencv.hpp"
#include "argparse.hpp"
#include "benchmark.hpp"
#include "utimer.hpp"

#include "processing/motion_detector.h"
#include "processing/motion_detector_stream.hpp"
#include "processing/motion_detector_buffer.hpp"
#include "processing/frame.hpp"

#define PROGRAM_VERSION "0.1"

cv::VideoCapture read_capture(const std::string &filename)
{
    cv::VideoCapture cap(filename);
    if (!cap.isOpened())
    {
        std::cerr << "Could not open video file: " << filename << std::endl;
        exit(1);
    }
    return cap;
}

int main(int argc, char const *argv[])
{
    // Define program arguments
    argparse::ArgumentParser parser("motion-detection", PROGRAM_VERSION);
    parser.add_description("Count the number of frames with motion w.r.t the first frame of the video.");

    parser.add_argument("-s", "--source-video").required().help("path of the source video");
    parser.add_argument("-t", "--threshold").help("threshold for the motion detection").default_value(0.6).scan<'g', double>();
    parser.add_argument("-w", "--workers").default_value(0).help("number of workers (0 for sequential)").scan<'i', int>();
    parser.add_argument("-m", "--parallel-mode").default_value(0).help("parallel mode (0: threads, 1: fast flow, 2: OpenMP)").scan<'i', int>();
    parser.add_argument("-pl", "--player").default_value(false).implicit_value(true).help("shows the video player with workers = 0 (ESC to exit)");
    parser.add_argument("-b", "--benchmark").default_value(std::string("")).help("benchmark mode is enabled and appends the results with the specified name in results.csv");
    parser.add_argument("-i", "--iterations").default_value(5).help("benchmark mode is executed with the specified number of iterations").scan<'i', int>();
    parser.add_argument("-p", "--prefetch").default_value(false).implicit_value(true).help("prefetch the frames by loading the entire video");

    // Parse arguments
    try
    {
        parser.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
        std::exit(1);
    }

    // Get arguments
    const auto source_path = parser.get<std::string>("source-video");
    const auto motion_detection_threshold = parser.get<double>("threshold");
    const auto workers = parser.get<int>("workers");
    const auto parallel_mode = parser.get<int>("parallel-mode");
    const auto show_video = parser.get<bool>("player");
    const auto benchmark_name = parser.get<std::string>("benchmark");
    const auto benchmark_iterations = parser.get<int>("iterations");
    const auto prefetch = parser.get<bool>("prefetch");

    // Detect motion
    auto cap = read_capture(source_path);
    unsigned long frames_with_motion = 0;

#if BLUR == 1
    std::cout << "Blur algorithm: H1" << std::endl;
#elif BLUR == 2
    std::cout << "Blur algorithm: H2" << std::endl;
#elif BLUR == 3
    std::cout << "Blur algorithm: H3" << std::endl;
#elif BLUR == 4
    std::cout << "Blur algorithm: H4" << std::endl;
#elif BLUR == 5
    std::cout << "Blur algorithm: OPEN_CV" << std::endl;
#else
    std::cout << "Blur algorithm: Box blur" << std::endl;
#endif

#if OPENCV_GREYSCALE == 1
    std::cout << "OpenCV greyscale algorithm: true" << std::endl;
#else
    std::cout << "OpenCV greyscale algorithm: false" << std::endl;
#endif

    video::IMotionDetector *motion_detector;
    if (prefetch)
    {
        motion_detector = new video::MotionDetectorBuffer(cap, motion_detection_threshold);
    }
    else
    {
        motion_detector = new video::MotionDetectorStream(cap, motion_detection_threshold);
    }

    if (!benchmark_name.empty() && benchmark_iterations >= 1)
    {
        unsigned long total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
        std::cout << "Benchmark mode enabled" << std::endl;
        std::cout << "Iterations: " << benchmark_iterations << std::endl;
        std::cout << "Results will be appended to results.csv" << std::endl;
        for (int it = 1; it <= benchmark_iterations; it++)
        {
            std::cout << "\nStarting benchmark " << it << " for " << benchmark_name << std::endl;
            auto count_frames = [&]() -> unsigned long
            { return motion_detector->count_frames(); };
            auto count_frames_threads = [&](int workers) -> unsigned long
            { return motion_detector->count_frames_threads(workers); };
            auto count_frames_ff = [&](int workers) -> unsigned long
            { return motion_detector->count_frames_ff(workers); };
            auto count_frames_omp = [&](int workers) -> unsigned long
            { return motion_detector->count_frames_omp(workers); };
            helper::benchmark(benchmark_name, it, count_frames, std::vector<std::function<unsigned long(int)>>{count_frames_threads, count_frames_ff, count_frames_omp}, std::vector<std::string>{"threads", "ff", "omp"}, total_frames);
        }
    }
    else
    {
        if (workers == 0)
        {
            if (show_video)
            {
                frames_with_motion = motion_detector->count_frames_player();
            }
            else
            {
                {
                    helper::utimer timer("Counting frames sequential");
                    frames_with_motion = motion_detector->count_frames();
                }
            }
        }
        else
        {
            if (parallel_mode == 0)
            {
                std::cout << "Parallel mode: threads" << std::endl;
                {
                    helper::utimer timer("Counting frames threads");
                    frames_with_motion = motion_detector->count_frames_threads(workers);
                }
            }
            else if (parallel_mode == 1)
            {
                std::cout << "Parallel mode: fast flow" << std::endl;
                {
                    helper::utimer timer("Counting frames fast flow");
                    frames_with_motion = motion_detector->count_frames_ff(workers);
                }
            }
            else
            {
                std::cout << "Parallel mode: OpenMP" << std::endl;
                {
                    helper::utimer timer("Counting frames OpenMP");
                    frames_with_motion = motion_detector->count_frames_omp(workers);
                }
            }
        }

        // Print the results
        ulong total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
        float percentage = (float)frames_with_motion / (float)total_frames * 100.0f;
        std::cout << "The number of frames with motion are " << frames_with_motion << "/" << total_frames << " (" << percentage << "%)" << std::endl;
    }

    delete motion_detector;
    cap.release();
    return 0;
}
