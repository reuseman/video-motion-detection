#include <iostream>
#include "opencv2/opencv.hpp"
#include "processing.hpp"
#include "argparse.hpp"
#include "benchmark.hpp"
#include "utimer.hpp"

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
    parser.add_argument("-w", "--workers").default_value(0).help("number of workers (0 for sequential)");
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
            assert(false);
    }();
    auto show_video = parser.get<bool>("player");
    auto benchmark_name = parser.get<std::string>("benchmark");
    auto benchmark_iterations = parser.get<int>("iterations");
    auto verbose = parser.get<bool>("verbose");

    // TODO remove this hardcoded path
    source_path = "./assets/door.mov";
    auto cap = read_capture(source_path);

    // Detect motion
    int frames_with_motion = 0;
    if (!benchmark_name.empty() && benchmark_iterations >= 1)
    {
        std::cout << "Benchmark mode enabled" << std::endl;
        std::cout << "Iterations: " << benchmark_iterations << std::endl;
        std::cout << "Results will be appended to results.csv" << std::endl;
        for (int it = 1; it <= benchmark_iterations; it++)
        {
            std::cout << "\nStarting benchmark " << it << " for " << benchmark_name << std::endl;
            helper::benchmark(benchmark_name, it, video::count_frames_with_motion, video::count_frames_with_motion_par, cap, opencv_greyscale, blur_algorithm, motion_detection_threshold, verbose);
        }
    }
    else
    {
        if (workers == 0)
        {
            if (show_video)
            {
                frames_with_motion = video::count_frames_with_motion_player(cap, opencv_greyscale, blur_algorithm, motion_detection_threshold, verbose);
            }
            else
            {
                frames_with_motion = video::count_frames_with_motion(cap, opencv_greyscale, blur_algorithm, motion_detection_threshold, verbose);
            }
        }
        else
        {
            frames_with_motion = video::count_frames_with_motion_par(cap, opencv_greyscale, blur_algorithm, motion_detection_threshold, workers, verbose);
        }
    }

    // Print the results
    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    float percentage = (float)frames_with_motion / (float)total_frames * 100.0f;
    std::cout << "The number of frames with motion are " << frames_with_motion << "/" << total_frames << " (" << percentage << "%)" << std::endl;
    cap.release();
    return 0;
}