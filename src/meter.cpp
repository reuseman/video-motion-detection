#include <iostream>
#include <functional>

#include "opencv2/opencv.hpp"
#include "frame_processing.hpp"
#include "argparse.hpp"
#include "benchmark.hpp"
#include "utimer.hpp"
#include "timer.hpp"
#include "motion_detector.h"
#include "motion_detector_stream.hpp"
#include "motion_detector_buffer.hpp"
#include "shared_queue.hpp"
#include <queue>

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
    parser.add_argument("-o", "--opencv-greyscale").default_value(false).implicit_value(true).help("use opencv greyscale");
    parser.add_argument("-a", "--blur-algorithm").default_value(std::string("BOX_BLUR")).help("blur algorithms (H1, H2, H3, H4, BOX_BLUR, BOX_BLUR_MOVING_WINDOW, OPEN_CV)");
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
    const auto prefetch = parser.get<bool>("prefetch");

    // Detect motion

    int64_t total_read_time;
    int64_t frame_read_time;

    int64_t total_deque_time;
    int64_t frame_deque_time;

    int64_t total_grey_time;
    int64_t frame_grey_time;

    int64_t total_blur_time;
    int64_t frame_blur_time;

    int64_t total_detect_motion_time;
    int64_t frame_detect_motion_time;

    helper::ChronoTimer timer;

    // Read time
    timer.reset();
    auto cap = read_capture(source_path);
    auto frames_number = cap.get(cv::CAP_PROP_FRAME_COUNT);
    auto iterations = 10;

    for (int i = 0; i < iterations; i++)
    {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        std::queue<cv::Mat> frames;
        cv::Mat background;
        cap >> background;
        while (true)
        {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty())
                break;
            frames.push(frame);
        }
        total_read_time = timer.elapsed_us();
        
        video::frame::preprocess(background);
        cv::Mat frame;
        while (!frames.empty())
        {
            timer.reset();
            frame = frames.front();
            frames.pop();
            total_deque_time += timer.elapsed_us();

            // Grey time
            timer.reset();
            video::frame::grayscale(frame);
            total_grey_time += timer.elapsed_us();

            // Blur time
            timer.reset();
            video::frame::box_blur(frame, 3);
            total_blur_time += timer.elapsed_us();

            // Motion detection
            timer.reset();
            cv::Mat diff;
            absdiff(background, frame, diff);
            int count = countNonZero(diff);
            float difference_percentage = (float)count / (float)(diff.rows * diff.cols);
            bool motion = difference_percentage > motion_detection_threshold;
            total_detect_motion_time += timer.elapsed_us();
        }
    }

    // compute the average time
    frame_read_time = total_read_time / (frames_number * iterations);
    frame_deque_time = total_deque_time / (frames_number * iterations);
    frame_grey_time = total_grey_time / (frames_number * iterations);
    frame_blur_time = total_blur_time / (frames_number * iterations);
    frame_detect_motion_time = total_detect_motion_time / (frames_number * iterations);

    // Print results
    std::cout << "Average time per frame" << std::endl;
    std::cout << "Read time: " << frame_read_time << " us" << std::endl;
    std::cout << "Deque time: " << frame_deque_time << " us" << std::endl;
    std::cout << "Grey time: " << frame_grey_time << " us" << std::endl;
    std::cout << "Blur time: " << frame_blur_time << " us" << std::endl;
    std::cout << "Motion detection time: " << frame_detect_motion_time << " us" << std::endl;
    std::cout << "Total time: " << frame_read_time + frame_deque_time + frame_grey_time + frame_blur_time + frame_detect_motion_time << " us" << std::endl;

    // Print each single total time
    std::cout << "\nTotal time" << std::endl;
    std::cout << "Total Read time: " << total_read_time << " us" << std::endl;
    std::cout << "Total Deque time: " << total_deque_time << " us" << std::endl;
    std::cout << "Total Grey time: " << total_grey_time << " us" << std::endl;
    std::cout << "Total Blur time: " << total_blur_time << " us" << std::endl;
    std::cout << "Total Motion detection time: " << total_detect_motion_time << " us" << std::endl;
    std::cout << "Total time: " << total_read_time + total_deque_time + total_grey_time + total_blur_time + total_detect_motion_time << " us" << std::endl;

    cap.release();
    return 0;
}


// FILE: house720.mp4
// Average time per frame
// Read time: 469 us
// Deque time: 0 us
// Grey time: 1404 us
// Blur time: 6430 us
// Motion detection time: 326 us
// Total time: 8629 us

// Total time
// Total Read time: 394369 us
// Total Deque time: 578 us
// Total Grey time: 1179595 us
// Total Blur time: 5401349 us
// Total Motion detection time: 274033 us
// Total time: 7249924 us

// FILE: door1080.mov
// Average time per frame
// Read time: 819 us
// Deque time: 1 us
// Grey time: 3304 us
// Blur time: 15014 us
// Motion detection time: 836 us
// Total time: 19974 us                 // It's proportional

// Total time
// Total Read time: 7577425 us
// Total Deque time: 9536 us
// Total Grey time: 30564264 us
// Total Blur time: 138884206 us
// Total Motion detection time: 7734803 us
// Total time: 184770234 us