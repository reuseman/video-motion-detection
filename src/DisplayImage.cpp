#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "utimer.hpp"
#include "argparse.hpp"

#define PROGRAM_VERSION "0.1"

// Convulution kernels
cv::Mat h1 = cv::Mat::ones(3, 3, CV_8U);
cv::Mat h2 = (cv::Mat_<short>(3, 3) << 1, 1, 1, 1, 2, 1, 1, 1, 1);
cv::Mat h3 = (cv::Mat_<short>(3, 3) << 1, 2, 1, 2, 4, 2, 1, 2, 1);
cv::Mat h4 = (cv::Mat_<short>(3, 3) << 1, 1, 1, 1, 0, 1, 1, 1, 1);

void print_frame_stats(cv::Mat frame, std::string message)
{   
    std::cout << "Frame stats - " << message << std::endl;
    std::cout << "Number of channels: " << frame.channels() << std::endl;
    std::cout << "Depth: " << frame.depth() << std::endl;
    std::cout << "Size of frame: " << frame.size() << std::endl;
    std::cout << "Total number of pixels: " << frame.total() << std::endl;
    std::cout << "Size of frame in bytes: " << frame.step << std::endl;
}

void cv_grayscale(cv::Mat& frame)
{
    // Take the 3 channels and compute the average
    cv::Mat bgr[3];
    split(frame, bgr);
    cv::Mat avg = (bgr[0] + bgr[1] + bgr[2]) / 3;
    // Convert to 8 bit
    avg.convertTo(avg, CV_8U);
    // Update frame with the average, i.e. grayscale
    avg.copyTo(frame);
}

void cv_apply_kernel(cv::Mat& frame, cv::Mat kernel)
{
    // TODO handle the edge values with a mirroring scheme
    assert (kernel.rows == kernel.cols);
    assert (kernel.rows % 2 == 1);

    int radius = (kernel.rows - 1) / 2;
    int total_weight = cv::sum(kernel)[0];
    cv::Mat old_frame = frame.clone();

    for (int row = radius; row < frame.rows - radius; row++)
    {
        for (int col = radius; col < frame.cols - radius; col++)
        {
            int sum = 0;
            for (int k = -radius; k <= radius; k++)
            {
                for (int l = -radius; l <= radius; l++)
                    sum += old_frame.at<uchar>(row + k, col + l) * kernel.at<double>(k + radius, l + radius);
            }
            frame.at<uchar>(row, col) = sum / total_weight;
        }
    }
}

void cv_box_blur(cv::Mat& frame, int kernel_size)
{   
    assert (kernel_size % 2 == 1);
    int radius = (kernel_size - 1) / 2;

    // Vertical pass
    cv::Mat old_frame = frame.clone();
    for (int row = radius; row < frame.rows - radius; row++)
    {
        for (int col = radius; col < frame.cols - radius; col++)
        {
            int sum = 0;
            for (int k = -radius; k <= radius; k++)
            {
                sum += old_frame.at<uchar>(row + k, col);
            }
            frame.at<uchar>(row, col) = sum / kernel_size;
        }
    }

    // Horizontal pass
    old_frame = frame.clone();
    for (int col = radius; col < frame.cols - radius; col++)
    {
        for (int row = radius; row < frame.rows - radius; row++)
        {
            int sum = 0;
            for (int k = -radius; k <= radius; k++)
            {
                sum += old_frame.at<uchar>(row, col + k);
            }
            frame.at<uchar>(row, col) = sum / kernel_size;
        }
    }
}

void cv_box_blur_moving_window(cv::Mat& frame, int kernel_size)
{
    assert (kernel_size % 2 == 1);
    int radius = (kernel_size - 1) / 2;

    std::queue<int> write_buffer;

    // First horizontal pass
    for (int row = 0; row < frame.rows; row++)
    {
        int sum = 0;
        // Initialize the write buffer with the left edge
        for (int col = 0; col < kernel_size; col++)
        {
            sum += frame.at<uchar>(row, col);
            // Edge handling without doing nothing
            if (col < radius)
                write_buffer.push(frame.at<uchar>(row, col));
        }
        write_buffer.push(sum / kernel_size);

        // Execute where the window can cover the whole kernel size
        for (int col = radius + 1; col < frame.cols - radius; col++)
        {   
            sum -= frame.at<uchar>(row, col - radius - 1);
            frame.at<uchar>(row, col - radius - 1) = write_buffer.front();
            write_buffer.pop();
            sum += frame.at<uchar>(row, col + radius);
            write_buffer.push(sum / kernel_size);
        }

        // Handle the other edge
        for (int col = frame.cols - kernel_size; col < frame.cols; col++)
        {
            if (col < frame.cols - radius)
            {
                frame.at<uchar>(row, col) = write_buffer.front();
                write_buffer.pop();
            }
            else
            {
                frame.at<uchar>(row, col) = frame.at<uchar>(row, col);
            }
        }
    }

    // First vertical pass
    for (int col = 0; col < frame.cols; col++)
    {
        int sum = 0;
        // Initialize the write buffer with the left edge
        for (int row = 0; row < kernel_size; row++)
        {
            sum += frame.at<uchar>(row, col);
            // Edge handling without doing nothing
            if (row < radius)
                write_buffer.push(frame.at<uchar>(row, col));
        }
        write_buffer.push(sum / kernel_size);

        // Execute where the window can cover the whole kernel size
        for (int row = radius + 1; row < frame.rows - radius; row++)
        {   
            sum -= frame.at<uchar>(row - radius - 1, col);
            frame.at<uchar>(row - radius - 1, col) = write_buffer.front();
            write_buffer.pop();
            sum += frame.at<uchar>(row + radius, col);
            write_buffer.push(sum / kernel_size);
        }

        // Handle the other edge
        for (int row = frame.rows - kernel_size; row < frame.rows; row++)
        {
            if (row < frame.rows - radius)
            {
                frame.at<uchar>(row, col) = write_buffer.front();
                write_buffer.pop();
            }
            else
            {
                frame.at<uchar>(row, col) = frame.at<uchar>(row, col);
            }
        }
    }
}

void cv_preprocess(cv::Mat& frame)
{
    // Apply greyscale
    // cvtColor(frame, frame, COLOR_BGR2GRAY);
    cv_grayscale(frame);

    // Apply a box blur (a.k.a. box linear filter)
    cv_box_blur_moving_window(frame, 3);
}

bool frame_contains_motion(cv::Mat background_frame, cv::Mat& current_frame, float motion_detection_threshold)
{
    cv::Mat diff;
    cv_preprocess(current_frame);
    absdiff(background_frame, current_frame, diff);
    // threshold(diff, diff, motion_detection_threshold, 255, THRESH_BINARY);

    // Count the number of pixels that are above the threshold
    int count = countNonZero(diff);
    float difference_percentage = (float)count / (float)(diff.rows * diff.cols);
    return difference_percentage > motion_detection_threshold;
}

int main(int argc, char const *argv[])
{
    // Arguments parsing
    argparse::ArgumentParser parser("Motion video detect", PROGRAM_VERSION);
    parser.add_description("Count the number of frames with motion w.r.t the first frame of the video.");
    // Positional arguments
    parser.add_argument("input").help("path of the input video");
    // Optional arguments
    parser.add_argument("-t", "--threshold").help("sets the threshold for motion detection").default_value(0.6).scan<'g', double>();
    parser.add_argument("-p", "--player").default_value(false).implicit_value(true).help("shows the video player (ESC to exit)");
    parser.add_argument("--verbose").default_value(false).implicit_value(true).help("verbose mode");

    try 
    {
        parser.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
        std::exit(1);
    }

    auto input_path = parser.get<std::string>("input");
    auto show_video = parser.get<bool>("player");
    auto verbose = parser.get<bool>("verbose");
    auto motion_detection_threshold = parser.get<double>("threshold");

    // Create a VideoCapture object and open the input file
    cv::VideoCapture cap("../assets/video.mp4"); 
   
    // Check if video opened successfully
    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }


    cv::Mat background_frame;
    cv::Mat frame;
    cap >> background_frame;
    frame = background_frame;

    if (verbose)
        print_frame_stats(background_frame, "before preprocessing");

    // Detect motion
    helper::utimer u("Video motion detect");
    int frames_with_motion = 0;
    int current_frame = 0;
    int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    cv_preprocess(background_frame);

    if (show_video)
    {
        while (current_frame < total_frames)
        {
            if (frame_contains_motion(background_frame, frame, motion_detection_threshold)) {
                frames_with_motion++;
                if (verbose)
                    std::cout << "Frame " << current_frame + 1 << "/" << total_frames << " has motion" <<  std::endl;

                // Add back the RGB channels to show a red border of 1% of the height
                cv::Mat channels[3] = {frame, frame, frame};
                cv::merge(channels, 3, frame);
                cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols, frame.rows), cv::Scalar(0, 0, 255), frame.size().height * 0.01);
            }
            cv::imshow("Video", frame);
            if (cv::waitKey(30) == 27)
                break;

            cap >> frame;
            current_frame++;
        }

        cv::destroyAllWindows();
    }
    else
    {
        while (current_frame < total_frames)
        {
            if (frame_contains_motion(background_frame, frame, motion_detection_threshold)) {
                frames_with_motion++;
                if (verbose)
                    std::cout << "Frame " << current_frame + 1<< "/" << total_frames << " has motion" <<  std::endl;
            }

            cap >> frame;
            current_frame++;
        }
    }

    float percentage = (float) frames_with_motion / (float) total_frames * 100.0f;
    std::cout << "The number of frames with motion are " << frames_with_motion << "/" << total_frames << " (" << percentage << "%)" << std::endl;
    
    // Release the video capture object and close the frame
    cap.release();
    return 0;
}