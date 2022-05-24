#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

namespace helper
{
    template <class T>
    class SharedQueue
    {
    private:
        std::queue<T> queue;
        std::mutex mutex;
        std::condition_variable queue_with_something;

    public:
        void push(T const &value)
        {
            {
                std::unique_lock<std::mutex> ul(mutex);
                queue.push(value);
            }
            queue_with_something.notify_one();
        }

        T pop()
        {
            std::unique_lock<std::mutex> ul(mutex);
            queue_with_something.wait(ul, [this]
                                      { return !queue.empty(); });
            T value = queue.front();
            queue.pop();
            return value;
        }

        bool empty()
        {
            std::unique_lock<std::mutex> ul(mutex);
            return queue.empty();
        }

        void clear()
        {
            std::unique_lock<std::mutex> ul(mutex);
        }
    };
}