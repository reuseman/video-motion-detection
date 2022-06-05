#pragma once
#include <thread>
#include <atomic>
#include "opencv2/opencv.hpp"
#include "shared_queue.hpp"

namespace video::frame
{
    // Convulution kernels
    cv::Mat blur_h1 = cv::Mat::ones(3, 3, CV_8U);
    cv::Mat blur_h1_7 = cv::Mat::ones(7, 7, CV_8U);
    cv::Mat blur_h2 = (cv::Mat_<short>(3, 3) << 1, 1, 1, 1, 2, 1, 1, 1, 1);
    cv::Mat blur_h3 = (cv::Mat_<short>(3, 3) << 1, 2, 1, 2, 4, 2, 1, 2, 1);
    cv::Mat blur_h4 = (cv::Mat_<short>(3, 3) << 1, 1, 1, 1, 0, 1, 1, 1, 1);

    void print_stats(cv::Mat frame, std::string message)
    {
        std::cout << "Frame stats - " << message << std::endl;
        std::cout << "Number of channels: " << frame.channels() << std::endl;
        std::cout << "Depth: " << frame.depth() << std::endl;
        std::cout << "Size of frame: " << frame.size() << std::endl;
        std::cout << "Total number of pixels: " << frame.total() << std::endl;
        std::cout << "Size of frame in bytes: " << frame.step << std::endl;
    }

    // Complexity O(nm)
    void grayscale(cv::Mat &frame)
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

    // O(mn)
    void apply_kernel(cv::Mat &frame, cv::Mat kernel)
    {
        assert(kernel.rows == kernel.cols);
        assert(kernel.rows % 2 == 1);

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
                        sum += old_frame.at<uchar>(row + k, col + l) * kernel.at<short>(k + radius, l + radius);
                }
                frame.at<uchar>(row, col) = sum / total_weight;
            }
        }
    }

    void box_blur(cv::Mat &frame, int kernel_size)
    {
        assert(kernel_size % 2 == 1);
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

    void preprocess(cv::Mat &frame)
    {

// Apply greyscale
#if OPENCV_GREYSCALE
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
#else
        grayscale(frame);
#endif

        // Apply blur
#if BLUR == 1
        apply_kernel(frame, blur_h1);
#elif BLUR == 2
        apply_kernel(frame, blur_h2);
#elif BLUR == 3
        apply_kernel(frame, blur_h3);
#elif BLUR == 4
        apply_kernel(frame, blur_h4);
#elif BLUR == 5
        cv::blur(frame, frame, cv::Size(3, 3));
#elif BLUR == 7
        apply_kernel(frame, blur_h1_7);
#else
        box_blur(frame, 3);
#endif
    }

    // Complexity O(nm)
    bool contains_motion(cv::Mat background_frame, cv::Mat current_frame, float motion_detection_threshold)
    {
        cv::Mat diff;
        preprocess(current_frame);

        absdiff(background_frame, current_frame, diff);

        // Count the number of pixels that are above the threshold
        int count = countNonZero(diff);
        float difference_percentage = (float)count / (float)(diff.rows * diff.cols);
        return difference_percentage > motion_detection_threshold;
    }

    void add_red_border(cv::Mat &frame)
    {
        // Add back the RGB channels to show a red border of 1% of the height
        cv::Mat channels[3] = {frame, frame, frame};
        cv::merge(channels, 3, frame);
        cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols, frame.rows), cv::Scalar(0, 0, 255), frame.size().height * 0.01);
    }

} // namespace video::frame
