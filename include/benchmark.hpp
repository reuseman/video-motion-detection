#pragma once
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <iomanip>
#include <functional>
#include <chrono>

using namespace std;

namespace helper
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
      ofstrm << "trial_ame,id,type,threads,completion_time,bandwidth,speedup,scalability,efficiency" << endl;
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
      auto us = measure<>::duration(g, args..., i).count();
      cout << "Par time with " << i << " cores: " << us << " us" << endl;
      ofstrm << trial_name << "," << id << ",par, " << i << "," << us << "," << (double)1 / us << "," << (double)seq_time / us << "," << (double)par_time_1 / us << "," << (double)(seq_time / i) / us << endl;
    }
    ofstrm.close();
  }

  template <class F, class G, class... Args>
  void benchmark(string trial_name, int id, F &&f, std::vector<G> &&g, std::vector<std::string> names, unsigned long items, Args &&...args)
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
      ofstrm << "trial name,items,id,type,threads,completion time,service time,bandwidth,speedup,scalability,efficiency" << endl;
    }

    // BENCHMARK
    const auto processor_count = std::thread::hardware_concurrency();

    // Test the serial function
    auto seq_time = measure<>::duration(f, args...).count();
    cout << "Seq time: " << seq_time << " us" << endl;

    // Write results to file
    ofstrm << std::fixed << setprecision(2) << endl;
    ofstrm << trial_name << ","  << items << "," << id << ",seq, 1," << seq_time << "," << 1 << "," << 1 << "," << 1 << "," << 1 << "," << 1 << endl;

    for (int i = 0; i < g.size(); i++)
    {
      G g_i = g[i];
      std::string name = names[i];
      auto par_time_1 = measure<>::duration(g_i, args..., 1).count();
      cout << "Parallel benchmark of " << name << endl;

      std::vector<int> workers = {1};
      for (int j = 2; j <= processor_count; j = 2 + j)
      {
        workers.push_back(j);
      }

      // Test parallel function with j cores
      // for (int j = 1; j <= processor_count; j = 2 * j)
      for (auto j : workers)     
      {
        auto us = measure<>::duration(g_i, args..., j).count();
        cout << "Par time with " << j << " cores: " << us << " us" << endl;
        auto completion_time = us;
        auto service_time = us/items;
        auto bandwith = (double)1 / service_time;
        auto speedup = (double)seq_time / completion_time;
        auto scalability = (double)par_time_1 / completion_time;
        auto efficiency = (double)(seq_time / j) / completion_time;
        ofstrm << trial_name << ","  << items << "," << id << "," << name << "," << j << "," << completion_time << "," << service_time << "," << bandwith << "," << speedup << "," << scalability << "," << efficiency << endl;
      }
    }
    ofstrm.close();
  }
}