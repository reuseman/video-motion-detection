#include <iostream>
#include <chrono>

#include "opencv2/opencv.hpp"
#include "argparse.hpp"
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
    argparse::ArgumentParser parser("motion-detection-meter", PROGRAM_VERSION);
    parser.add_description("Measure the time for the different type of computations");

    parser.add_argument("-s", "--source-video").required().help("path of the source video");
    parser.add_argument("-t", "--threshold").help("threshold for the motion detection").default_value(0.6).scan<'g', double>();
    parser.add_argument("-p", "--prefetch").default_value(false).implicit_value(true).help("prefetch the frames by loading the entire video");
    parser.add_argument("-i", "--iterations").default_value(5).help("benchmark mode is executed with the specified number of iterations").scan<'i', int>();

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
    const auto prefetch = parser.get<bool>("prefetch");
    const auto iterations = parser.get<int>("iterations");

    using clock = std::chrono::high_resolution_clock;
    std::chrono::microseconds total_read_time{0};
    std::chrono::microseconds total_deque_time{0};
    std::chrono::microseconds total_grey_time{0};
    std::chrono::microseconds total_blur_time{0};
    std::chrono::microseconds total_detect_motion_time{0};
    std::chrono::microseconds entire_program{0};

    auto cap = read_capture(source_path);
    int frames_number = cap.get(cv::CAP_PROP_FRAME_COUNT);

    for (int i = 0; i < iterations; ++i)
    {
        std::cout << "Iteration " << i + 1 << std::endl;
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        std::queue<cv::Mat> frames;
        cv::Mat background;
        cap >> background;
        video::frame::preprocess(background);

        int counter = 0;

        while (true)
        {
            auto start = clock::now();
            cv::Mat frame;
            cap >> frame;
            total_read_time += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start);
            if (frame.empty())
                break;

            // Grey time
            start = clock::now();
            video::frame::grayscale(frame);
            total_grey_time += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start);

            // Blur time
            start = clock::now();
            // video::frame::box_blur(frame, 3);
            video::frame::apply_kernel(frame, video::frame::blur_h1);
            // cv::blur(frame, frame, cv::Size(3, 3));
            total_blur_time += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start);

            // Motion detection
            start = clock::now();
            bool motion = video::frame::difference_bigger_than_threshold(background, frame, motion_detection_threshold);
            total_detect_motion_time += std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - start);
        }
    }

    cap.release();

    // Compute the average and print results
    auto frame_read_time = (total_read_time / (frames_number * iterations)).count();
    auto frame_deque_time = (total_deque_time / (frames_number * iterations)).count();
    auto frame_grey_time = (total_grey_time / (frames_number * iterations)).count();
    auto frame_blur_time = (total_blur_time / (frames_number * iterations)).count();
    auto frame_detect_motion_time = (total_detect_motion_time / (frames_number * iterations)).count();
    auto frame_entire_time = (entire_program / (frames_number * iterations)).count();

    std::cout << "\nAverage time per frame" << std::endl;
    std::cout << "Read time: " << frame_read_time << " us" << std::endl;
    std::cout << "Deque time: " << frame_deque_time << " us" << std::endl;
    std::cout << "Grey time: " << frame_grey_time << " us" << std::endl;
    std::cout << "Blur time: " << frame_blur_time << " us" << std::endl;
    std::cout << "Motion detection time: " << frame_detect_motion_time << " us" << std::endl;
    std::cout << "Summed time: " << frame_read_time + frame_deque_time + frame_grey_time + frame_blur_time + frame_detect_motion_time << " us" << std::endl;
    std::cout << "Total time: " << frame_entire_time << " us" << std::endl;

    std::cout << "\nOverall time" << std::endl;
    std::cout << "Read time: " << total_read_time.count() << " us" << std::endl;
    std::cout << "Deque time: " << total_deque_time.count() << " us" << std::endl;
    std::cout << "Grey time: " << total_grey_time.count() << " us" << std::endl;
    std::cout << "Blur time: " << total_blur_time.count() << " us" << std::endl;
    std::cout << "Motion detection time: " << total_detect_motion_time.count() << " us" << std::endl;
    std::cout << "Total time: " << total_read_time.count() + total_deque_time.count() + total_grey_time.count() + total_blur_time.count() + total_detect_motion_time.count() << " us" << std::endl;

    return 0;
}