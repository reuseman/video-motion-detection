#pragma once
#include "ff/ff.hpp"
#include "opencv2/opencv.hpp"
#include "frame.hpp"
#include <vector>
#include <thread>

typedef struct
{
    cv::Mat *frame;
    bool contains_motion;
    int counter;
} task;

class ff_emitter_stream : public ff::ff_monode_t<task>
{
private:
    cv::VideoCapture *cap;
    int total_frames = 0;
    int counter = 0;

public:
    ff_emitter_stream(cv::VideoCapture *cap) : cap(cap) {
        this->total_frames = cap->get(cv::CAP_PROP_FRAME_COUNT);
    }

    task *svc(task *t)
    {
        for (int i = 1; i < total_frames; i++)
        {
            cv::Mat *frame = new cv::Mat();
            cap->read(*frame);
            if (frame->empty())
            {
                delete frame;
                break;
            }
            ff_send_out(new task{frame, false, i});
        }
        return (EOS);
    }
};

class ff_emitter_buffer : public ff::ff_monode_t<task>
{
private:
    std::vector<cv::Mat> *frames;

public:
    ff_emitter_buffer(std::vector<cv::Mat> *frames) : frames(frames) {}

    task *svc(task *t)
    {
        for (int i = 1; i < frames->size(); i++)
        {
            cv::Mat *frame = new cv::Mat();
            *frame = frames->at(i);
            if (frame->empty())
            {
                delete frame;
                break;
            }
            ff_send_out(new task{frame, false, i});
        }
        return (EOS);
    }
};

class ff_worker : public ff::ff_node_t<task>
{
private:
    cv::Mat *background;
    float threshold;

public:
    ff_worker(cv::Mat *background, float threshold) : background(background), threshold(threshold) {}

    task *svc(task *t)
    {
        t->contains_motion = video::frame::contains_motion(*(background), *(t->frame), this->threshold);
        return t;
    }
};

class ff_worker_acc : public ff::ff_node_t<task>
{
private:
    cv::Mat *background;
    float threshold;
    ulong local_counter = 0;
    std::atomic<ulong> *global_counter;

public:
    ff_worker_acc(cv::Mat *background, float threshold, std::atomic<ulong> *global_counter) : background(background), threshold(threshold), global_counter(global_counter) {}

    task *svc(task *t)
    {
        if (video::frame::contains_motion(*(background), *(t->frame), this->threshold))
            local_counter++;
        delete t->frame;
        delete t;
        return GO_ON;
    }

    void svc_end()
    {
        global_counter->fetch_add(local_counter);
    }
};


class ff_grey : public ff::ff_monode_t<task>
{
private:

public:
    task *svc(task *t)
    {
        video::frame::grayscale(*(t->frame));
        return t;
    }
};

class ff_blur : public ff::ff_node_t<task>
{
private:

public:
    task *svc(task *t)
    {
        video::frame::apply_blur(*(t->frame));
        return t;
    }
};

class ff_motion : public ff::ff_minode_t<task>
{
private:
    cv::Mat *background;
    float threshold;

public:
    ff_motion(cv::Mat *background, float threshold) : background(background), threshold(threshold) {}

    task *svc(task *t)
    {
        t->contains_motion = video::frame::difference_bigger_than_threshold(*(background), *(t->frame), this->threshold);
        return t;
    }
};

class ff_collector : public ff::ff_minode_t<task>
{
private:
    ulong *frames_with_motion = 0;

public:
    ff_collector(ulong *frames_with_motion) : frames_with_motion(frames_with_motion) {}

    task *svc(task *t)
    {
        if (t->contains_motion)
            *frames_with_motion += 1;
        delete t->frame;
        delete t;
        return (GO_ON);
    }
};