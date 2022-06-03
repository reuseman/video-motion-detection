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
        virtual ulong count_frames_ff(int workers) = 0;
        virtual ulong count_frames_omp(int workers) = 0;
    };

}