#pragma once
#include <iostream>
// #include <ff/ff.hpp>
#include <functional>

#include "opencv2/opencv.hpp"
#include "frame_processing.hpp"
#include "argparse.hpp"
#include "benchmark.hpp"
#include "utimer.hpp"
#include "motion_detector.hpp"

#define PROGRAM_VERSION "0.1"

typedef unsigned long ulong;

using namespace std::literals::chrono_literals;

const auto ta = 100ms;
const auto tf = 100ms;
const auto m = 100;

typedef struct
{
    int taskno;
    float x;
} myTask;

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

// struct source : ff::ff_node_t<cv::VideoCapture, cv::Mat>
// {
//     source(const cv::VideoCapture cap) : cap(cap) {}

//     cv::Mat *svc(cv::VideoCapture *)
//     {
//         while (true)
//         {
//             cv::Mat *frame = new cv::Mat();
//             cap >> *frame; // maybe clone it
//             if (frame->empty())
//             {
//                 delete frame;
//                 return (EOS);
//             }
//             std::this_thread::sleep_for(ta);
//             ff_send_out(frame);
//         }
//     }

//     cv::VideoCapture cap;
// };

// struct funstageF : ff::ff_node_t<cv::Mat, bool>
// {
//     funstageF(cv::Mat background, float motion_detection_threshold, bool opencv_greyscale, video::frame::BlurAlgorithm blur_algorithm) : background(background), motion_detection_threshold(motion_detection_threshold), opencv_greyscale(opencv_greyscale), blur_algorithm(blur_algorithm) {}

//     bool *svc(cv::Mat *frame)
//     {
//         bool *motion = new bool(video::frame::contains_motion(background, *frame, motion_detection_threshold, opencv_greyscale, blur_algorithm));
//         delete frame;
//         return motion;
//     }

//     const cv::Mat background;
//     const float motion_detection_threshold;
//     const bool opencv_greyscale;
//     const video::frame::BlurAlgorithm blur_algorithm;
// };

// struct sink : ff::ff_node_t<bool>
// {
//     bool *svc(bool *motion)
//     {
//         if (*motion) {
//             std::cout << "Motion detected!" << std::endl;
//             counter++;
//         } else {
//             std::cout << "No motion detected." << std::endl;
//         }
//         delete motion;
//         return (GO_ON);
//     }

//     void svc_end()
//     {
//         std::cout << "Sink got EOS, total sum = " << counter << std::endl;
//     }

//     long counter = 0;
// };

// ulong count_frames_ff(cv::VideoCapture cap, bool opencv_greyscale, video::frame::BlurAlgorithm blur_algorithm, float motion_detection_threshold, bool verbose, int workers)
// {
//     cv::Mat background;
//     cap >> background;
//     video::frame::preprocess(background, opencv_greyscale, blur_algorithm);

//     std::vector<std::unique_ptr<ff::ff_node>> workers_nodes;
//     for (int i = 0; i < workers; i++)
//     {
//         workers_nodes.push_back(std::make_unique<funstageF>(background, motion_detection_threshold, opencv_greyscale, blur_algorithm));
//     }
//     std::cout << "Workers created" << workers_nodes.size() << std::endl;
//     ff::ff_Farm<myTask> farm(std::move(workers_nodes));

//     source emitter(cap);
//     sink collector;

//     farm.add_emitter(emitter);
//     farm.add_collector(collector);

//     ff::ffTime(ff::START_TIME);
//     if (farm.run_and_wait_end() < 0)
//     {
//         ff::error("Error running farm");
//         return -1;
//     }
//     ff::ffTime(ff::STOP_TIME);
//     std::cout << "Farm time: " << ff::ffTime(ff::GET_TIME) << std::endl;
//     return 42L;
// }

int main(int argc, char const *argv[])
{
    // Define program arguments
    argparse::ArgumentParser parser("motion-detection", PROGRAM_VERSION);
    parser.add_description("Count the number of frames with motion w.r.t the first frame of the video.");

    parser.add_argument("-s", "--source-video").required().help("path of the source video");
    parser.add_argument("-t", "--threshold").help("threshold for the motion detection").default_value(0.6).scan<'g', double>();
    parser.add_argument("-w", "--workers").default_value(0).help("number of workers (0 for sequential)").scan<'i', int>();
    parser.add_argument("-m", "--parallel-mode").default_value(0).help("parallel mode (0: threads, 1: fast flow)").scan<'i', int>();
    parser.add_argument("-o", "--opencv-greyscale").default_value(false).implicit_value(true).help("use opencv greyscale");
    parser.add_argument("-a", "--blur-algorithm").default_value(std::string("BOX_BLUR_MOVING_WINDOW")).help("blur algorithms (H1, H2, H3, H4, BOX_BLUR, BOX_BLUR_MOVING_WINDOW, OPEN_CV)");
    parser.add_argument("-p", "--player").default_value(false).implicit_value(true).help("shows the video player with workers = 0 (ESC to exit)");
    parser.add_argument("-b", "--benchmark").default_value(std::string("")).help("benchmark mode is enabled and appends the results with the specified name in results.csv");
    parser.add_argument("-i", "--iterations").default_value(1).help("benchmark mode is executed with the specified number of iterations").scan<'i', int>();
    parser.add_argument("--verbose").default_value(false).implicit_value(true).help("verbose mode");

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
    auto source_path = parser.get<std::string>("source-video");
    auto motion_detection_threshold = parser.get<double>("threshold");
    auto workers = parser.get<int>("workers");
    auto parallel_mode = parser.get<int>("parallel-mode");
    auto opencv_greyscale = parser.get<bool>("opencv-greyscale");
    auto blur_algorithm = [=]() -> video::frame::BlurAlgorithm
    {
        std::string algorithm = parser.get<std::string>("blur-algorithm");
        if (algorithm == "H1")
            return video::frame::H1;
        else if (algorithm == "H2")
            return video::frame::H2;
        else if (algorithm == "H3")
            return video::frame::H3;
        else if (algorithm == "H4")
            return video::frame::H4;
        else if (algorithm == "BOX_BLUR")
            return video::frame::BOX_BLUR;
        else if (algorithm == "BOX_BLUR_MOVING_WINDOW")
            return video::frame::BOX_BLUR_MOVING_WINDOW;
        else if (algorithm == "OPEN_CV")
            return video::frame::OPEN_CV;
        else
            throw std::runtime_error("Unknown blur algorithm");
    }();
    auto show_video = parser.get<bool>("player");
    auto benchmark_name = parser.get<std::string>("benchmark");
    auto benchmark_iterations = parser.get<int>("iterations");
    auto verbose = parser.get<bool>("verbose");

    // Detect motion
    auto cap = read_capture(source_path);
    unsigned long frames_with_motion = 0;
    video::MotionDetector motion_detector(cap, motion_detection_threshold, opencv_greyscale, blur_algorithm, verbose);

    if (!benchmark_name.empty() && benchmark_iterations >= 1)
    {
        std::cout << "Benchmark mode enabled" << std::endl;
        std::cout << "Iterations: " << benchmark_iterations << std::endl;
        std::cout << "Results will be appended to results.csv" << std::endl;
        for (int it = 1; it <= benchmark_iterations; it++)
        {
            std::cout << "\nStarting benchmark " << it << " for " << benchmark_name << std::endl;
            auto count_frames = [&]() -> unsigned long
            { return motion_detector.count_frames(); };
            auto count_frames_without_motion_threads = [&](int workers) -> unsigned long
            { return motion_detector.count_frames_threads(workers); };
            helper::benchmark(benchmark_name, it, count_frames, count_frames_without_motion_threads);
        }
    }
    else
    {
        if (workers == 0)
        {
            if (show_video)
            {
                frames_with_motion = motion_detector.count_frames_player();
            }
            else
            {
                frames_with_motion = motion_detector.count_frames();
            }
        }
        else
        {
            if (parallel_mode == 0)
            {
                std::cout << "Parallel mode: threads" << std::endl;
                frames_with_motion = motion_detector.count_frames_threads(workers);
            }
            else
            {
                std::cout << "Parallel mode: fast flow" << std::endl;
                // frames_with_motion = count_frames_ff(cap, opencv_greyscale, blur_algorithm, motion_detection_threshold, verbose, workers);
            }
        }
    }

    // Print the results
    ulong total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    float percentage = (float)frames_with_motion / (float)total_frames * 100.0f;
    std::cout << "The number of frames with motion are " << frames_with_motion << "/" << total_frames << " (" << percentage << "%)" << std::endl;
    cap.release();
    return 0;
}