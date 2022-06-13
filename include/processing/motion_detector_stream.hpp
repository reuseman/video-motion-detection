#pragma once

#include "ff/ff.hpp"
#include "fastflow_nodes.hpp"

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

    public:
        MotionDetectorStream(cv::VideoCapture &cap, const float threshold) : cap(cap), threshold(threshold) {}
        virtual ulong count_frames();
        virtual ulong count_frames_player();
        virtual ulong count_frames_threads(int workers);
        virtual ulong count_frames_threads_pinned(int workers);
        virtual ulong count_frames_ff(int workers);
        virtual ulong count_frames_ff_acc(int workers);
        virtual ulong count_frames_ff_on_demand(int workers);
        virtual ulong count_frames_ff_pipe_farm(int workers);
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
                break;
            }

            queue.push(frame);
        }

        // Wait for threads to finish
        for (auto &t : threads)
            t.join();

        return frames_with_motion;
    }

    ulong MotionDetectorStream::count_frames_threads_pinned(int workers)
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
            int current_core = sched_getcpu();
            int old_core = sched_getcpu();

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
            stick_thread_to_core(&threads.back(), i);
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
            workers_nodes.push_back(std::make_unique<ff_worker>(&background, this->threshold));

        // Create the farm
        ulong frames_with_motion = 0;
        ff_emitter_stream emitter(&this->cap);
        ff_collector collector(&frames_with_motion);
        ff::ff_Farm<task> farm(std::move(workers_nodes), emitter, collector);

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

    ulong MotionDetectorStream::count_frames_ff_acc(int workers)
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
        ff::ff_Farm<task> farm(std::move(workers_nodes));
        ff_emitter_stream emitter(&this->cap);
        farm.add_emitter(emitter);
        farm.remove_collector();

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

    ulong MotionDetectorStream::count_frames_ff_on_demand(int workers)
    {
        ulong frames_with_motion = 0;

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
        ff::ff_Farm<task> farm(std::move(workers_nodes));
        ff_emitter_stream emitter(&this->cap);
        ff_collector collector(&frames_with_motion);
        farm.add_emitter(emitter);
        farm.add_collector(collector);
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

    ulong MotionDetectorStream::count_frames_ff_pipe_farm(int workers)
    {
        ulong frames_with_motion = 0;

        // Get the preprocessed background
        this->cap.set(cv::CAP_PROP_POS_FRAMES, 0); // Go to the beginning of the video
        cv::Mat background;
        this->cap >> background;
        video::frame::preprocess(background);

        // Initialize workers for the ff_Farm
        ff::ff_pipeline p;
        std::vector<std::unique_ptr<ff::ff_node>> workers_blur;

        for (int i = 0; i < workers; i++)
        {
            workers_blur.push_back(std::make_unique<ff_blur>());
        }

        // Create the farms
        ff_emitter_stream emitter(&this->cap);
        ff_grey grey;
        ff::ff_Farm<task> farm_blur(std::move(workers_blur));
        ff_motion ff_motion_node(&background, this->threshold);
        ff_collector collector(&frames_with_motion);

        farm_blur.add_emitter(grey);

        // Create the farm
        p.add_stage(std::move(emitter));
        p.add_stage(&farm_blur);
        farm_blur.add_collector(ff_motion_node);
        p.add_stage(&collector);

        if (p.run_and_wait_end() < 0)
        {
            ff::error("Error running pipe of farms");
            return -1;
        }

#if MOTION_VERBOSE
        p.ffStats(std::cout);
#endif
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
