#include <chrono>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

constexpr int   kMatrixSize          = 512;
constexpr float kComparisonTolerance = 1e-3f;

__global__ void MatrixMultiplyKernel(const float* matrix_a,
                                     const float* matrix_b,
                                           float* result_matrix,
                                           int    matrix_width) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < matrix_width && col < matrix_width) {
        float sum = 0.0f;
        for (int k = 0; k < matrix_width; ++k) {
            sum += matrix_a[row * matrix_width + k] * matrix_b[k * matrix_width + col];
        }
        result_matrix[row * matrix_width + col] = sum;
        
        if (row == 0 && col == 0) {
            printf("CUDA kernel executed successfully. First element sum = %f\n", sum);
        }
    }
}

void MatrixMultiplyCpu(const float* matrix_a,
                       const float* matrix_b,
                             float* result_matrix,
                             int    matrix_width) {
    for (int row = 0; row < matrix_width; ++row) {
        for (int col = 0; col < matrix_width; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < matrix_width; ++k) {
                sum += matrix_a[row * matrix_width + k] * matrix_b[k * matrix_width + col];
            }

            result_matrix[row * matrix_width + col] = sum;
        }
    }
}

bool CompareResults(const float* cpu_result,
                    const float* gpu_result,
                          int element_count) {
    for (int i = 0; i < element_count; ++i) {
        if (fabs(cpu_result[i] - gpu_result[i]) > kComparisonTolerance) {
            return false;
        }
    }
    return true;
}

int main() {
    const int matrix_element_count = kMatrixSize * kMatrixSize;
    const size_t matrix_bytes = matrix_element_count * sizeof(float);

    // Allocate host memory
    float* host_matrix_a   = new float[matrix_element_count];
    float* host_matrix_b   = new float[matrix_element_count];
    float* host_result_cpu = new float[matrix_element_count];
    float* host_result_gpu = new float[matrix_element_count];

    // Initialize matrices
    for (int i = 0; i < matrix_element_count; ++i) {
        host_matrix_a[i] = 1.0f;
        host_matrix_b[i] = 2.0f;
    }

    // Allocate device memory
    float* device_matrix_a = nullptr;
    float* device_matrix_b = nullptr;
    float* device_result   = nullptr;
    
    cudaMalloc(&device_matrix_a, matrix_bytes);
    cudaMalloc(&device_matrix_b, matrix_bytes);
    cudaMalloc(&device_result,   matrix_bytes);

    // Copy data to device
    cudaMemcpy(device_matrix_a, host_matrix_a, matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_matrix_b, host_matrix_b, matrix_bytes, cudaMemcpyHostToDevice);

    // GPU execution timing
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start);

    // Launch kernel
    const dim3 block_dimensions(16, 16);
    const dim3 grid_dimensions(
        (kMatrixSize + block_dimensions.x - 1) / block_dimensions.x,
        (kMatrixSize + block_dimensions.y - 1) / block_dimensions.y
    );
    
    MatrixMultiplyKernel<<<grid_dimensions, block_dimensions>>>(
        device_matrix_a, device_matrix_b, device_result, kMatrixSize);
    cudaDeviceSynchronize();

    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, gpu_start, gpu_stop);

    cudaMemcpy(host_result_gpu, device_result, matrix_bytes, cudaMemcpyDeviceToHost);

    const auto cpu_start_time = std::chrono::high_resolution_clock::now();
    MatrixMultiplyCpu(host_matrix_a, host_matrix_b, host_result_cpu, kMatrixSize);
    const auto cpu_end_time = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> cpu_duration = cpu_end_time - cpu_start_time;

    const bool results_match = CompareResults(host_result_cpu, host_result_gpu, matrix_element_count);
    std::cout << (results_match ? "Results match!" : "Results do not match!") << std::endl;
    std::cout << "GPU result[0] = " << host_result_gpu[0] 
              << ", CPU result[0] = " << host_result_cpu[0] << std::endl;
    std::cout << "GPU execution time: " << gpu_time_ms / 1000.0 << " seconds" << std::endl;
    std::cout << "CPU execution time: " << cpu_duration.count() << " seconds" << std::endl;

    cudaFree(device_matrix_a);
    cudaFree(device_matrix_b);
    cudaFree(device_result);
    
    delete[] host_matrix_a;
    delete[] host_matrix_b;
    delete[] host_result_cpu;
    delete[] host_result_gpu;

    return 0;
}