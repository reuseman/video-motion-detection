#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <functional>
#include <chrono>

using namespace std;

namespace utils
{
  template <class TimeT = std::chrono::microseconds,
            class ClockT = std::chrono::steady_clock>
  struct measure
  {
    template <class F, class... Args>
    static auto duration(F &&func, Args &&...args)
    {
      auto start = ClockT::now();
      std::invoke(std::forward<F>(func), std::forward<Args>(args)...);
      return std::chrono::duration_cast<TimeT>(ClockT::now() - start);
    }
  };

  template <class F, class G, class... Args>
  void benchmark(string trial_name, int id, F &&f, G &&g, Args &&...args)
  {
    // Check if file exists
    std::ios_base::openmode mode;
    ifstream ifstrm("results.csv");
    if (ifstrm.good())
    {
      mode = std::ios_base::app;
    }
    else
    {
      mode = std::ios_base::out;
    }

    // Open file
    ofstream ofstrm("results.csv", mode);
    if (mode == std::ios_base::out)
    {
      ofstrm << "trial name,id,type,cores,completion time,bandwidth,speedup,scalability,efficiency" << endl;
    }

    // BENCHMARK
    const auto processor_count = std::thread::hardware_concurrency();

    // Test the serial function
    auto seq_time = measure<>::duration(f, args...).count();
    cout << "Seq time: " << seq_time << " us" << endl;

    // Write results to file
    ofstrm << std::fixed << setprecision(2) << endl;
    ofstrm << trial_name << "," << id << ",seq, 1," << seq_time << "," << 1 << "," << 1 << "," << 1 << "," << 1 << endl;

    auto par_time_1 = measure<>::duration(g, args..., 1).count();
    for (int i = 1; i <= processor_count; i = 2 * i)
    {
      // Test parallel function with i cores
      cout << "Parallel translation with " << i << " threads" << endl;
      auto ms = measure<>::duration(g, args..., i).count();
      cout << "Par time with " << i << " cores: " << ms << " us" << endl;
      ofstrm << trial_name << "," << id << ",par, " << i << "," << ms << "," << (double)1 / ms << "," << (double)seq_time / ms << "," << (double)par_time_1 / ms << "," << (double)(seq_time / i) / ms << endl;
    }
    ofstrm.close();
  }

  template <class F, class G, class... Args>
  void benchmark(string trial_name, int id, F &&f, std::vector<G> &&g, std::vector<std::string> names, Args &&...args)
  {
    // Check if file exists
    std::ios_base::openmode mode;
    ifstream ifstrm("results.csv");
    if (ifstrm.good())
    {
      mode = std::ios_base::app;
    }
    else
    {
      mode = std::ios_base::out;
    }

    // Open file
    ofstream ofstrm("results.csv", mode);
    if (mode == std::ios_base::out)
    {
      ofstrm << "trial name,id,type,cores,completion time,bandwidth,speedup,scalability,efficiency" << endl;
    }

    // BENCHMARK
    const auto processor_count = std::thread::hardware_concurrency();

    // Test the serial function
    auto seq_time = measure<>::duration(f, args...).count();
    cout << "Seq time: " << seq_time << " us" << endl;

    // Write results to file
    ofstrm << std::fixed << setprecision(2) << endl;
    ofstrm << trial_name << "," << id << ",seq, 1," << seq_time << "," << 1 << "," << 1 << "," << 1 << "," << 1 << endl;

    for (int i = 0; i < g.size(); i++)
    {
      G g_i = g[i];
      std::string name = names[i];
      auto par_time_1 = measure<>::duration(g_i, args..., 1).count();
      cout << "Parallel benchmark of " << name << endl;
      // Test parallel function with j cores
      for (int j = 1; j <= processor_count; j = 2 * j)
      {
        // cout << "Parallel translation with " << j << " threads" << endl;
        auto ms = measure<>::duration(g_i, args..., j).count();
        // cout << "Par time with " << j << " cores: " << ms << " us" << endl;
        ofstrm << trial_name << "," << id << "," << name << "," << j << "," << ms << "," << (double)1 / ms << "," << (double)seq_time / ms << "," << (double)par_time_1 / ms << "," << (double)(seq_time / j) / ms << endl;
      }
    }
    ofstrm.close();
  }
}