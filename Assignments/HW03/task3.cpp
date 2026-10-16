#include "msort.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <ratio>
#include <vector>

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " n t ts\n";
    return 1;
  }

  const std::size_t n = std::strtoull(argv[1], nullptr, 10);
  const int t = std::atoi(argv[2]);
  const std::size_t ts = std::strtoull(argv[3], nullptr, 10);
  std::vector<int> arr(n);

  std::random_device entropy_source;
  std::mt19937 generator(entropy_source());
  std::uniform_int_distribution<int> dist(-1000, 1000);
  for (std::size_t i = 0; i < n; ++i) {
    arr[i] = dist(generator);
  }

  omp_set_num_threads(t);

  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_ms;

  start = high_resolution_clock::now();
  msort(arr.data(), n, ts);
  end = high_resolution_clock::now();

  duration_ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

  cout << arr.front() << '\n' << arr.back() << '\n' << duration_ms.count() << '\n';
  return 0;
}
