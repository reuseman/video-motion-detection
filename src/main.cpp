#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "utimer.hpp"
#include "argparse.hpp"
#include "processing.hpp"
#include <atomic>

#define PROGRAM_VERSION "0.1"

int main(int argc, char const *argv[])
{
    // Define program arguments
    argparse::ArgumentParser parser("Motion video detect", PROGRAM_VERSION);
    parser.add_description("Count the number of frames with motion w.r.t the first frame of the video.");
    //      Positional arguments
    parser.add_argument("input").help("path of the input video");
    //      Optional arguments
    parser.add_argument("-t", "--threshold").help("sets the threshold for motion detection").default_value(0.6).scan<'g', double>();
    parser.add_argument("-w", "--workers").default_value(0).help("number of workers (0 for sequential)");
    parser.add_argument("-o", "--opencv-greyscale").default_value(false).implicit_value(true).help("use opencv greyscale");
    parser.add_argument("-b", "--blur-algorithm").default_value(std::string("BOX_BLUR_MOVING_WINDOW")).help("blur algorithms (H1, H2, H3, H4, BOX_BLUR, BOX_BLUR_MOVING_WINDOW, OPEN_CV)");
    parser.add_argument("-p", "--player").default_value(false).implicit_value(true).help("shows the video player (ESC to exit)");
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
    auto input_path = parser.get<std::string>("input");
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
    auto verbose = parser.get<bool>("verbose");

    // Create a VideoCapture object and open the input file
    cv::VideoCapture cap("./assets/video2.mov");
    if (!cap.isOpened())
    {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    // Detect motion
    int frames_with_motion = 0;
    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);

    helper::utimer u("Video motion detect");
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

    float percentage = (float)frames_with_motion / (float)total_frames * 100.0f;
    std::cout << "The number of frames with motion are " << frames_with_motion << "/" << total_frames << " (" << percentage << "%)" << std::endl;

    // Release the VideoCapture object
    cap.release();
    return 0;
}