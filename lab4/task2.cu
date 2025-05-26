#include <chrono>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

namespace {
    constexpr int kImageWidth  = 1024;
    constexpr int kImageHeight = 1024;
    constexpr int kImageSize   = kImageWidth * kImageHeight;
    constexpr int kBlurRadius  = 1;
    constexpr int kBlurArea    = (2 * kBlurRadius + 1) * (2 * kBlurRadius + 1);
}

__global__ void ApplyBlurGpu(const unsigned char* input_image,
                                   unsigned char* blurred_image,
                                   int            width,
                                   int            height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < kBlurRadius || y < kBlurRadius || 
        x >= width - kBlurRadius || y >= height - kBlurRadius) {
        return;
    }

    int pixel_sum = 0;
    for (int dy = -kBlurRadius; dy <= kBlurRadius; ++dy) {
        for (int dx = -kBlurRadius; dx <= kBlurRadius; ++dx) {
            const int neighbor_x = x + dx;
            const int neighbor_y = y + dy;
            pixel_sum += input_image[neighbor_y * width + neighbor_x];
        }
    }

    blurred_image[y * width + x] = static_cast<unsigned char>(pixel_sum / kBlurArea);
}

void ApplyBlurCpu(const unsigned char* input_image,
                        unsigned char* blurred_image,
                        int            width,
                        int            height) {
    for (int y = kBlurRadius; y < height - kBlurRadius; ++y) {
        for (int x = kBlurRadius; x < width - kBlurRadius; ++x) {
            int pixel_sum = 0;
            for (int dy = -kBlurRadius; dy <= kBlurRadius; ++dy) {
                for (int dx = -kBlurRadius; dx <= kBlurRadius; ++dx) {
                    pixel_sum += input_image[(y + dy) * width + (x + dx)];
                }
            }

            blurred_image[y * width + x] = static_cast<unsigned char>(pixel_sum / kBlurArea);
        }
    }
}

bool CompareImages(const unsigned char* image_a,
                   const unsigned char* image_b,
                         int            pixel_count) {
    for (int i = 0; i < pixel_count; ++i) {
        if (image_a[i] != image_b[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    // Allocate and initialize host memory
    unsigned char* host_input_image = new unsigned char[kImageSize];
    unsigned char* host_blurred_cpu = new unsigned char[kImageSize];
    unsigned char* host_blurred_gpu = new unsigned char[kImageSize];

    // Fill input image with random values
    for (int i = 0; i < kImageSize; ++i) {
        host_input_image[i] = static_cast<unsigned char>(rand() % 256);
    }

    // Process on CPU and measure time
    const auto cpu_start_time = std::chrono::high_resolution_clock::now();
    ApplyBlurCpu(host_input_image, host_blurred_cpu, kImageWidth, kImageHeight);
    const auto cpu_end_time = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> cpu_duration = cpu_end_time - cpu_start_time;

    // Allocate device memory
    unsigned char* device_input_image = nullptr;
    unsigned char* device_blurred_image = nullptr;
    cudaMalloc(&device_input_image, kImageSize);
    cudaMalloc(&device_blurred_image, kImageSize);

    // Copy input to device
    cudaMemcpy(device_input_image, host_input_image, kImageSize, cudaMemcpyHostToDevice);

    // Configure and launch GPU kernel
    const dim3 block_dimensions(16, 16);
    const dim3 grid_dimensions(
        (kImageWidth + block_dimensions.x - 1) / block_dimensions.x,
        (kImageHeight + block_dimensions.y - 1) / block_dimensions.y
    );

    // Time GPU execution
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start);

    ApplyBlurGpu<<<grid_dimensions, block_dimensions>>>(
        device_input_image, device_blurred_image, kImageWidth, kImageHeight);
    cudaDeviceSynchronize();

    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, gpu_start, gpu_stop);

    // Copy results back to host
    cudaMemcpy(host_blurred_gpu, device_blurred_image, kImageSize, cudaMemcpyDeviceToHost);

    // Verify results
    const bool results_match = CompareImages(host_blurred_cpu, host_blurred_gpu, kImageSize);
    std::cout << (results_match ? "Results match!" : "Results do not match!") << '\n';
    std::cout << "Input pixel[512] = " << static_cast<int>(host_input_image[512]) << '\n';
    std::cout << "GPU blurred[512] = " << static_cast<int>(host_blurred_gpu[512]) << '\n';
    std::cout << "CPU blurred[512] = " << static_cast<int>(host_blurred_cpu[512]) << '\n';
    std::cout << "GPU execution time: " << gpu_time_ms / 1000.0 << " seconds\n";
    std::cout << "CPU execution time: " << cpu_duration.count() << " seconds\n";

    // Cleanup
    cudaFree(device_input_image);
    cudaFree(device_blurred_image);
    delete[] host_input_image;
    delete[] host_blurred_cpu;
    delete[] host_blurred_gpu;

    return 0;
}