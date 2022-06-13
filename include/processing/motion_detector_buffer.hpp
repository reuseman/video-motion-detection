#pragma once

#include "ff/ff.hpp"
#include "fastflow_nodes.hpp"

#include "opencv2/opencv.hpp"
#include "shared_queue.hpp"
#include "motion_detector.h"
#include "frame.hpp"

namespace video
{
    typedef unsigned long ulong;

    class MotionDetectorBuffer : public IMotionDetector
    {
    private:
        cv::VideoCapture cap;
        float threshold;
        helper::SharedQueue<cv::Mat> queue;
        std::vector<cv::Mat> frames;

    public:
        MotionDetectorBuffer(cv::VideoCapture &cap, const float threshold) : cap(cap), threshold(threshold)
        {
            do
            {
                cv::Mat frame;
                cap >> frame;
                if (frame.empty())
                {
                    break;
                }
                frames.push_back(frame);
            } while (true);
        }
        ulong count_frames();
        ulong count_frames_player();
        ulong count_frames_threads(int workers);
        ulong count_frames_threads_pinned(int workers);
        ulong count_frames_parallel_for(int workers);
        ulong count_frames_ff(int workers);
        ulong count_frames_ff_acc(int workers);
        ulong count_frames_ff_on_demand(int workers);
        ulong count_frames_ff_pipe_farm(int workers);
        ulong count_frames_omp(int workers);
    };

    ulong MotionDetectorBuffer::count_frames()
    {
        cv::Mat background_frame, frame;
        ulong frames_with_motion = 0;
        int total_frames = frames.size();
#if MOTION_VERBOSE
        int current_frame = 1;
#endif

        background_frame = frames[0];
        video::frame::preprocess(background_frame);

        for (int i = 1; i < frames.size(); i++)
        {
            frame = frames[i];
            if (video::frame::contains_motion(background_frame, frame, this->threshold))
            {
                frames_with_motion++;
#if MOTION_VERBOSE
                std::cout << "Motion detected in frame " << current_frame << " of " << total_frames << std::endl;
#endif
            }
#if MOTION_VERBOSE
            current_frame++;
#endif
        }

        return frames_with_motion;
    }

    ulong MotionDetectorBuffer::count_frames_player()
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

    ulong MotionDetectorBuffer::count_frames_threads(int workers)
    {
        helper::SharedQueue<cv::Mat> queue;
        std::vector<std::thread> threads;
        std::atomic<ulong> frames_with_motion = {0};

        cv::Mat background_frame = frames[0];
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

                        if (video::frame::contains_motion(background_frame, frame, this->threshold))
                            local_counter++;
                } }));
        }

        // Push frames to the queue
        for (int i = 1; i < frames.size(); i++)
        {
            queue.push(frames[i]);
        }
        for (int i = 0; i < workers; i++)
        {
            // push and empty frame with all zeros to signal EOS
            queue.push(cv::Mat());
        }

        // Wait for threads to finish
        for (auto &t : threads)
            t.join();

        return frames_with_motion;
    }

    ulong MotionDetectorBuffer::count_frames_threads_pinned(int workers)
    {
        return 0;
    }

    ulong MotionDetectorBuffer::count_frames_ff(int workers)
    {
        this->cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Go to the beginning of the video
        // Get the preprocessed background
        cv::Mat background;
        this->cap >> background;
        video::frame::preprocess(background);

        // Initialize workers for the ff_Farm
        std::vector<std::unique_ptr<ff::ff_node>> workers_nodes;
        for (int i = 0; i < workers; i++)
            workers_nodes.push_back(std::make_unique<ff_worker>(&background, this->threshold));

        // Create the farm
        ulong frames_with_motion = 0;
        ff::ff_Farm<cv::Mat, bool> farm(std::move(workers_nodes));
        ff_emitter_buffer emitter(&this->frames);
        ff_collector collector(&frames_with_motion);
        farm.add_emitter(emitter);
        farm.add_collector(collector);
        // farm.set_scheduling_ondemand();

        // ff::ffTime(ff::START_TIME);
        if (farm.run_and_wait_end() < 0)
        {
            ff::error("Error running farm");
            return -1;
        }
        // ff::ffTime(ff::STOP_TIME);
        // std::cout << "Farm time: " << ff::ffTime(ff::GET_TIME) << std::endl;
#if MOTION_VERBOSE
        farm.ffStats(std::cout);
#endif
        return frames_with_motion;
    }

    ulong MotionDetectorBuffer::count_frames_ff_acc(int workers)
    {
        std::atomic<ulong> frames_with_motion = {0};
        this->cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Go to the beginning of the video
        // Get the preprocessed background
        cv::Mat background;
        this->cap >> background;
        video::frame::preprocess(background);

        // Initialize workers for the ff_Farm
        std::vector<std::unique_ptr<ff::ff_node>> workers_nodes;
        for (int i = 0; i < workers; i++)
            workers_nodes.push_back(std::make_unique<ff_worker_acc>(&background, this->threshold, &frames_with_motion));

        // Create the farm
        ff::ff_Farm<cv::Mat, bool> farm(std::move(workers_nodes));
        ff_emitter_buffer emitter(&this->frames);
        farm.add_emitter(emitter);

        if (farm.run_and_wait_end() < 0)
        {
            ff::error("Error running farm");
            return -1;
        }
#if MOTION_VERBOSE
        farm.ffStats(std::cout);
#endif

        return frames_with_motion;
    }

    ulong MotionDetectorBuffer::count_frames_ff_on_demand(int workers)
    {
        std::atomic<ulong> frames_with_motion = {0};
        this->cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Go to the beginning of the video
        // Get the preprocessed background
        cv::Mat background;
        this->cap >> background;
        video::frame::preprocess(background);

        // Initialize workers for the ff_Farm
        std::vector<std::unique_ptr<ff::ff_node>> workers_nodes;
        for (int i = 0; i < workers; i++)
            workers_nodes.push_back(std::make_unique<ff_worker_acc>(&background, this->threshold, &frames_with_motion));

        // Create the farm
        ff::ff_Farm<cv::Mat, bool> farm(std::move(workers_nodes));
        ff_emitter_buffer emitter(&this->frames);
        farm.add_emitter(emitter);
        farm.set_scheduling_ondemand();

        if (farm.run_and_wait_end() < 0)
        {
            ff::error("Error running farm");
            return -1;
        }
#if MOTION_VERBOSE
        farm.ffStats(std::cout);
#endif
        return frames_with_motion;
    }

    ulong MotionDetectorBuffer::count_frames_ff_pipe_farm(int workers)
    {
        std::atomic<ulong> frames_with_motion = {0};
        return frames_with_motion;
    }

    ulong MotionDetectorBuffer::count_frames_omp(int workers)
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
        for (int i = 0; i < frames.size(); i++)
        {
            cv::Mat frame = frames[i];
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
