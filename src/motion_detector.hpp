// create class motion_detector
#pragma once
#include "opencv2/opencv.hpp"
#include "shared_queue.hpp"
#include "frame_processing.hpp"

namespace video
{
    typedef unsigned long ulong;

    class MotionDetector
    {
    private:
        cv::VideoCapture cap;
        float threshold;
        bool opencv_greyscale;
        video::frame::BlurAlgorithm blur_algorithm;
        helper::SharedQueue<cv::Mat> queue;

    public:
        MotionDetector(cv::VideoCapture cap, const float threshold, const bool opencv_greyscale, const video::frame::BlurAlgorithm blur_algorithm) : cap(cap), threshold(threshold), opencv_greyscale(opencv_greyscale), blur_algorithm(blur_algorithm) {}
        ulong count_frames();
        ulong count_frames_player();
        ulong count_frames_threads(int workers);
        ulong count_frames_ff(int workers);
    };

    ulong MotionDetector::count_frames()
    {
        cv::Mat background_frame, frame;
        ulong frames_with_motion = 0;
        int total_frames = this->cap.get(cv::CAP_PROP_FRAME_COUNT);
#if MOTION_VERBOSE
        int current_frame = 1;
#endif

        this->cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Go to the beginning of the video
        this->cap >> background_frame;
        video::frame::preprocess(background_frame, this->opencv_greyscale, this->blur_algorithm);

        cap >> frame;
        while (!frame.empty())
        {
            if (video::frame::contains_motion(background_frame, frame, this->threshold, this->opencv_greyscale, this->blur_algorithm))
            {
                frames_with_motion++;
#if MOTION_VERBOSE
                std::cout << "Motion detected in frame " << current_frame << " of " << total_frames << std::endl;
#endif
            }

            this->cap >> frame;
#if MOTION_VERBOSE
            current_frame++;
#endif
        }

        return frames_with_motion;
    }

    ulong MotionDetector::count_frames_player()
    {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Go to the beginning of the video
        cv::Mat background_frame;
        cv::Mat frame;
        cap >> background_frame;
        video::frame::preprocess(background_frame, opencv_greyscale, blur_algorithm);

        cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
        cv::setWindowProperty("Video", cv::WND_PROP_TOPMOST, cv::WINDOW_AUTOSIZE);
        ulong frames_with_motion = 0;

        cap >> frame;
        while (!frame.empty())
        {
            if (video::frame::contains_motion(background_frame, frame, this->threshold, this->opencv_greyscale, this->blur_algorithm))
            {
                frames_with_motion++;
                video::frame::add_red_border(frame);
            }

            cv::imshow("Video", frame);
            // Esc to exit
            if (cv::waitKey(1) == 27)
                break;

            cap >> frame;
        }
        return frames_with_motion;
    }

    ulong MotionDetector::count_frames_threads(int workers)
    {
        this->cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Go to the beginning of the video
        helper::SharedQueue<cv::Mat> queue;
        std::vector<std::thread> threads;
        std::atomic<ulong> frames_with_motion = {0};

        cv::Mat background_frame;
        this->cap >> background_frame;
        video::frame::preprocess(background_frame, this->opencv_greyscale, this->blur_algorithm);

#if MOTION_VERBOSE
        std::cout << "Starting " << workers << " threads" << std::endl;
#endif

        // Start threads
        for (int i = 0; i < workers; i++)
        {
            threads.push_back(std::thread([&]()
                                          {
            cv::Mat frame;
            ulong local_counter = 0;
            while (true)
            {
                frame = queue.pop();
                if (frame.empty()) {    // EOS reached, update global counter and exit
                    frames_with_motion += local_counter;
                    break;
                }

                if (video::frame::contains_motion(background_frame, frame,  this->threshold, this->opencv_greyscale, this->blur_algorithm))
                    local_counter++;
            } }));
        }

        // Push frames to the queue
        while (true)
        {
            cv::Mat frame;
            this->cap >> frame;
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
        for (auto &t : threads)
            t.join();

        return frames_with_motion;
    }

    ulong MotionDetector::count_frames_ff(int workers)
    {
        this->cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Go to the beginning of the video
        cv::Mat background_frame;
        cv::Mat frame;
        this->cap >> background_frame;
        frame = background_frame;

        ulong frames_with_motion = 0;
        int current_frame = 0;
        int total_frames = this->cap.get(cv::CAP_PROP_FRAME_COUNT);
        video::frame::preprocess(background_frame, this->opencv_greyscale, this->blur_algorithm);

        while (!frame.empty())
        {
            if (video::frame::contains_motion(background_frame, frame, this->threshold, this->opencv_greyscale, this->blur_algorithm))
            {
                frames_with_motion++;
            }

            this->cap >> frame;
            current_frame++;
        }

        return frames_with_motion;
    }
} // namespace video
