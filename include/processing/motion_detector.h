#pragma once
#include "opencv2/opencv.hpp"
#include "shared_queue.hpp"

typedef unsigned long ulong;

namespace video
{
    class IMotionDetector
    {

    public:
        virtual ~IMotionDetector() {}

        virtual ulong count_frames() = 0;
        virtual ulong count_frames_player() = 0;
        virtual ulong count_frames_threads(int workers) = 0;
        virtual ulong count_frames_threads_pinned(int workers) = 0;
        virtual ulong count_frames_ff(int workers) = 0;
        virtual ulong count_frames_ff_acc(int workers) = 0;
        virtual ulong count_frames_ff_on_demand(int workers) = 0;
        virtual ulong count_frames_ff_pipe_farm(int workers) = 0;
        virtual ulong count_frames_omp(int workers) = 0;

    protected:
        void stick_thread_to_core(std::thread *tid, int core)
        {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(core, &cpuset);
            if (pthread_setaffinity_np(tid->native_handle(), sizeof(cpu_set_t), &cpuset) != 0)
            {
                std::cerr << "Error setting thread affinity" << std::endl;
                exit(1);
            }
        }
    };

}