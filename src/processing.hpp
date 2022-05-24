#pragma once
#include <thread>
#include <atomic>
#include "opencv2/opencv.hpp"
#include "shared_queue.hpp"

namespace video::frame
{
    enum BlurAlgorithm
    {
        H1,
        H2,
        H3,
        H4,
        BOX_BLUR,
        BOX_BLUR_MOVING_WINDOW,
        OPEN_CV
    };

    // Convulution kernels
    cv::Mat blur_h1 = cv::Mat::ones(3, 3, CV_8U);
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

    // O(3n)
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
        // TODO handle the edge values with a mirroring scheme
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
                        sum += old_frame.at<uchar>(row + k, col + l) * kernel.at<double>(k + radius, l + radius);
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

    void box_blur_moving_window(cv::Mat &frame, int kernel_size)
    {
        assert(kernel_size % 2 == 1);
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

    void preprocess(cv::Mat &frame, bool opencv_greyscale, BlurAlgorithm blur_algorithm)
    {
        // Apply greyscale
        if (opencv_greyscale)
            cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        else
            grayscale(frame);

        // GREY 800
        // BLUR = 3500
        // MOTION = 100

        // Apply blur
        switch (blur_algorithm)
        {
        case H1:
            apply_kernel(frame, blur_h1);
            break;
        case H2:
            apply_kernel(frame, blur_h2);
            break;
        case H3:
            apply_kernel(frame, blur_h3);
            break;
        case H4:
            apply_kernel(frame, blur_h4);
            break;
        case BOX_BLUR:
            box_blur(frame, 3);
            break;
        case BOX_BLUR_MOVING_WINDOW:
            box_blur_moving_window(frame, 3);
            break;
        case OPEN_CV:
            cv::blur(frame, frame, cv::Size(3, 3));
            break;
        default:
            break;
        }
    }

    bool contains_motion(cv::Mat background_frame, cv::Mat &current_frame, float motion_detection_threshold, bool opencv_greyscale, BlurAlgorithm blur_algorithm)
    {
        cv::Mat diff;
        preprocess(current_frame, opencv_greyscale, blur_algorithm);

        absdiff(background_frame, current_frame, diff);
        // threshold(diff, diff, motion_detection_threshold, 255, THRESH_BINARY);

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

}

namespace video
{
    int count_frames_with_motion_player(cv::VideoCapture cap, bool opencv_greyscale, video::frame::BlurAlgorithm blur_algorithm, float motion_detection_threshold, bool verbose)
    {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Go to the beginning of the video
        cv::Mat background_frame;
        cv::Mat frame;
        cap >> background_frame;
        frame = background_frame;

        int frames_with_motion = 0;
        int current_frame = 0;
        int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
        video::frame::preprocess(background_frame, opencv_greyscale, blur_algorithm);

        cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
        cv::setWindowProperty("Video", cv::WND_PROP_TOPMOST, cv::WINDOW_AUTOSIZE);

        while (!frame.empty())
        {
            if (video::frame::contains_motion(background_frame, frame, motion_detection_threshold, opencv_greyscale, blur_algorithm))
            {
                frames_with_motion++;
                if (verbose)
                    std::cout << "Frame " << current_frame + 1 << "/" << total_frames << " has motion" << std::endl;

                video::frame::add_red_border(frame);
            }

            cv::imshow("Video", frame);
            // Esc to exit
            if (cv::waitKey(1) == 27)
                break;

            cap >> frame;
            current_frame++;
        }

        return frames_with_motion;
    }

    int count_frames_with_motion(cv::VideoCapture cap, bool opencv_greyscale, video::frame::BlurAlgorithm blur_algorithm, float motion_detection_threshold, bool verbose)
    {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Go to the beginning of the video
        cv::Mat background_frame;
        cv::Mat frame;
        cap >> background_frame;
        frame = background_frame;

        int frames_with_motion = 0;
        int current_frame = 0;
        int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
        video::frame::preprocess(background_frame, opencv_greyscale, blur_algorithm);

        while (!frame.empty())
        {
            if (video::frame::contains_motion(background_frame, frame, motion_detection_threshold, opencv_greyscale, blur_algorithm))
            {
                frames_with_motion++;
                if (verbose)
                    std::cout << "Frame " << current_frame + 1 << "/" << total_frames << " has motion" << std::endl;
            }

            cap >> frame;
            current_frame++;
        }

        return frames_with_motion;
    }

    int count_frames_with_motion_par(cv::VideoCapture cap, bool opencv_greyscale, video::frame::BlurAlgorithm blur_algorithm, float motion_detection_threshold, bool verbose, int workers)
    {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Go to the beginning of the video
        helper::SharedQueue<cv::Mat> queue;
        std::vector<std::thread> threads;
        std::atomic<int> counter;
        cv::Mat background_frame, frame;

        std::cout << "Starting " << workers << " threads" << std::endl;

        cap >> background_frame;
        frame = background_frame;
        video::frame::preprocess(background_frame, opencv_greyscale, blur_algorithm);
        counter = 0;

        int todo_total_frames = 0;

        // Start threads
        for (int i = 0; i < workers; i++)
        {
            threads.push_back(std::thread([&]()
                                          {
            cv::Mat frame;
            int local_counter = 0;
            while (true)
            {
                frame = queue.pop();
                if (frame.empty()) {
                    // EOS reached, update global counter and exit
                    counter += local_counter;
                    break;
                }

                if (video::frame::contains_motion(background_frame, frame,  motion_detection_threshold, opencv_greyscale, blur_algorithm))
                    local_counter++;

                todo_total_frames++;
            } }));
        }

        // Push frames to the queue
        while (true)
        {
            cap >> frame;
            if (frame.empty())
            {
                // Push empty frame to all threads to signal them the EOS
                for (auto &t : threads)
                    queue.push(frame);
                break;
            }

            queue.push(frame);
        }

        // Wait for threads to finish
        for (std::thread &t : threads)
            t.join();

        return counter;
    }

    int count_frames_with_motion_ff(cv::VideoCapture cap, bool opencv_greyscale, video::frame::BlurAlgorithm blur_algorithm, float motion_detection_threshold, bool verbose, int workers)
    {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Go to the beginning of the video
        cv::Mat background_frame;
        cv::Mat frame;
        cap >> background_frame;
        frame = background_frame;

        int frames_with_motion = 0;
        int current_frame = 0;
        int total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
        video::frame::preprocess(background_frame, opencv_greyscale, blur_algorithm);

        while (!frame.empty())
        {
            if (video::frame::contains_motion(background_frame, frame, motion_detection_threshold, opencv_greyscale, blur_algorithm))
            {
                frames_with_motion++;
                if (verbose)
                    std::cout << "Frame " << current_frame + 1 << "/" << total_frames << " has motion" << std::endl;
            }

            cap >> frame;
            current_frame++;
        }

        return frames_with_motion;
    }
}