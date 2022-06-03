#pragma once
#include <ff/ff.hpp>

#include "opencv2/opencv.hpp"
#include "shared_queue.hpp"
#include "frame.hpp"
#include "motion_detector.h"

namespace video
{
    typedef unsigned long ulong;

    class MotionDetectorStream : public IMotionDetector
    {
    private:
        cv::VideoCapture cap;
        float threshold;
        helper::SharedQueue<cv::Mat> queue;
        struct source : ff::ff_node_t<cv::VideoCapture, cv::Mat>
        {
            source(const cv::VideoCapture cap) : cap(cap) {}

            cv::Mat *svc(cv::VideoCapture *)
            {
                while (true)
                {
                    cv::Mat *frame = new cv::Mat();
                    cap >> *frame; // maybe clone it
                    if (frame->empty())
                    {
                        delete frame;
                        return (EOS);
                    }
                    // std::this_thread::sleep_for(ta);
                    ff_send_out(frame);
                }
            }

            cv::VideoCapture cap;
        };

        struct funstageF : ff::ff_node_t<cv::Mat, bool>
        {
            funstageF(cv::Mat background, const float motion_detection_threshold) : background(background), motion_detection_threshold(motion_detection_threshold) {}

            bool *svc(cv::Mat *frame)
            {
                bool *motion = new bool(video::frame::contains_motion(background, *frame, motion_detection_threshold));
                delete frame;
                return motion;
            }

            const cv::Mat background;
            const float motion_detection_threshold;
        };

        struct sink : ff::ff_node_t<bool>
        {

            sink(ulong *frames_with_motion) : frames_with_motion(frames_with_motion) {}

            bool *svc(bool *motion)
            {
                if (*motion)
                    (*frames_with_motion)++;

                delete motion;
                return (GO_ON);
            }

            ulong *frames_with_motion;
        };

    public:
        MotionDetectorStream(cv::VideoCapture cap, const float threshold) : cap(cap), threshold(threshold) {}
        virtual ulong count_frames();
        virtual ulong count_frames_player();
        virtual ulong count_frames_threads(int workers);
        virtual ulong count_frames_ff(int workers);
        virtual ulong count_frames_omp(int workers);
    };

    ulong MotionDetectorStream::count_frames()
    {
        cv::Mat background_frame, frame;
        ulong frames_with_motion = 0;
        int total_frames = this->cap.get(cv::CAP_PROP_FRAME_COUNT);
#if MOTION_VERBOSE
        int current_frame = 1;
#endif

        this->cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Go to the beginning of the video
        this->cap >> background_frame;
        video::frame::preprocess(background_frame);

        cap >> frame;
        while (!frame.empty())
        {
            if (video::frame::contains_motion(background_frame, frame, this->threshold))
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

    ulong MotionDetectorStream::count_frames_player()
    {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Go to the beginning of the video
        cv::Mat background_frame;
        cv::Mat frame;
        cap >> background_frame;
        video::frame::preprocess(background_frame);

        cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
        cv::setWindowProperty("Video", cv::WND_PROP_TOPMOST, cv::WINDOW_AUTOSIZE);
        ulong frames_with_motion = 0;

        cap >> frame;
        while (!frame.empty())
        {
            if (video::frame::contains_motion(background_frame, frame, this->threshold))
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

    ulong MotionDetectorStream::count_frames_threads(int workers)
    {
        this->cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Go to the beginning of the video
        helper::SharedQueue<cv::Mat> queue;
        std::vector<std::thread> threads;
        std::atomic<ulong> frames_with_motion = {0};

        cv::Mat background_frame;
        this->cap >> background_frame;
        video::frame::preprocess(background_frame);

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

                if (video::frame::contains_motion(background_frame, frame,  this->threshold))
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
                // std::cout << "finished to push frames" << std::endl;
                break;
            }

            queue.push(frame);
        }

        // Wait for threads to finish
        for (auto &t : threads)
            t.join();

        return frames_with_motion;
    }

    ulong MotionDetectorStream::count_frames_ff(int workers)
    {
        this->cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Go to the beginning of the video
        // Get the preprocessed background
        cv::Mat background;
        this->cap >> background;
        video::frame::preprocess(background);

        // Initialize workers for the ff_Farm
        std::vector<std::unique_ptr<ff::ff_node>> workers_nodes;
        for (int i = 0; i < workers; i++)
        {
            workers_nodes.push_back(std::make_unique<funstageF>(background, this->threshold));
        }

        // Create the farm
        ulong frames_with_motion = 0;
        ff::ff_Farm<cv::Mat, bool> farm(std::move(workers_nodes));
        source emitter(this->cap);
        sink collector(&frames_with_motion);
        farm.add_emitter(emitter);
        farm.add_collector(collector);

        // ff::ffTime(ff::START_TIME);
        if (farm.run_and_wait_end() < 0)
        {
            ff::error("Error running farm");
            return -1;
        }
        // ff::ffTime(ff::STOP_TIME);
        // std::cout << "Farm time: " << ff::ffTime(ff::GET_TIME) << std::endl;
        return frames_with_motion;
    }

    ulong MotionDetectorStream::count_frames_omp(int workers)
    {
        this->cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Go to the beginning of the video
        ulong frames_with_motion = 0;

        cv::Mat background_frame;
        this->cap >> background_frame;
        video::frame::preprocess(background_frame);

#if MOTION_VERBOSE
        std::cout << "Starting " << workers << " omp" << std::endl;
#endif

#pragma omp parallel num_threads(workers)
// Emitter
#pragma omp single
        while (true)
        {
            cv::Mat frame;
            this->cap >> frame;
            if (frame.empty())
                break;
#pragma omp task
            {
                if (video::frame::contains_motion(background_frame, frame, this->threshold))
#pragma omp atomic
                    frames_with_motion++;
            }
        }

        return frames_with_motion;
    }

} // namespace video
