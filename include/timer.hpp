#include <chrono>

namespace helper
{
    class ChronoTimer
    {
    public:
        std::chrono::time_point<std::chrono::high_resolution_clock> lastTime;
        ChronoTimer() : lastTime(std::chrono::high_resolution_clock::now()){};

        void reset()
        {
            lastTime = std::chrono::high_resolution_clock::now();
        }

        int64_t elapsed_us()
        {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - lastTime).count();
            return elapsed;
        };

        int64_t elapsed_ms()
        {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime).count();
            return elapsed;
        };

        int64_t elapsed_s()
        {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastTime).count();
            return elapsed;
        };
    };
}