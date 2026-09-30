#include "convolution.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <ratio>

int main(int, char *argv[]) {
  const std::size_t n = std::atoi(argv[1]);
  const std::size_t m = std::atoi(argv[2]);

  std::mt19937 generator(std::random_device{}());
  std::uniform_real_distribution<float> image_values(-10.0f, 10.0f);
  std::uniform_real_distribution<float> mask_values(-1.0f, 1.0f);

  float *image = new float[n * n];
  for (std::size_t i = 0; i < n * n; ++i) {
    image[i] = image_values(generator);
  }
  float *mask = new float[m * m];
  for (std::size_t i = 0; i < m * m; ++i) {
    mask[i] = mask_values(generator);
  }
  float *output = new float[n * n];

  const auto start = std::chrono::high_resolution_clock::now();
  convolve(image, output, n, mask, m);
  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          end - start);

  std::cout << elapsed.count() << '\n'
            << output[0] << '\n'
            << output[n * n - 1] << '\n';

  delete[] image;
  delete[] mask;
  delete[] output;
  return 0;
}
