#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/pair.h>
#include <nvtx3/nvToolsExt.h> 
#include <cub/cub.cuh>

__global__ void histogram_kernel(const unsigned char* __restrict__ in, int* hist, int n) {
    __shared__ int temp_hist[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x < 256){
        temp_hist[threadIdx.x] = 0;
    }
    __syncthreads();

    if (idx < n) {
        atomicAdd(&temp_hist[in[idx]], 1);
    }
    __syncthreads();

    if (threadIdx.x < 256) {
        atomicAdd(&hist[threadIdx.x], temp_hist[threadIdx.x]);
    }
}

__global__ void lut_kernel(const unsigned char* __restrict__ in, unsigned char* __restrict__ out, const unsigned char* __restrict__ lut, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = lut[in[idx]];
    }
}

void histogram_cub(const unsigned char* d_in, int* d_hist, int pixels) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_in, d_hist, 256 + 1, 0, 256, pixels);
    
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, d_in, d_hist, 256 + 1, 0, 256, pixels);
    
    cudaFree(d_temp_storage);
}

struct Normalize {
    int cdf_min;
    float range;
    
    __host__ __device__
    Normalize(int _min, float _range) : cdf_min(_min), range(_range) {}

    __host__ __device__
    unsigned char operator()(int cdf) const {
        float norm = ((float)cdf - cdf_min) / range * 255.0f;
        return (unsigned char)min(max(norm, 0.0f), 255.0f);
    }
};

int main() {
    nvtxRangePush("Full Application");

    cv::Mat img = cv::imread("images.png", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Nie znaleziono pliku images.png" << std::endl;
        return -1;
    }

    int pixels = img.rows * img.cols;

    thrust::device_vector<unsigned char> d_in(img.data, img.data + pixels);
    thrust::device_vector<unsigned char> d_out(pixels);
    thrust::device_vector<int> d_hist(256, 0);
    thrust::device_vector<int> d_cdf(256);
    thrust::device_vector<unsigned char> d_lut(256);

    int threads = 256;
    int blocks = (pixels + threads - 1) / threads;

    nvtxRangePush("Histogram Comparison");
    
    cudaDeviceSynchronize();
    auto start_kernel = std::chrono::high_resolution_clock::now();
    histogram_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_in.data()), 
        thrust::raw_pointer_cast(d_hist.data()), 
        pixels
    );
    cudaDeviceSynchronize();
    auto end_kernel = std::chrono::high_resolution_clock::now();
    
    thrust::fill(d_hist.begin(), d_hist.end(), 0);

    auto start_cub = std::chrono::high_resolution_clock::now();
    histogram_cub(
        thrust::raw_pointer_cast(d_in.data()), 
        thrust::raw_pointer_cast(d_hist.data()), 
        pixels
    );
    cudaDeviceSynchronize();
    auto end_cub = std::chrono::high_resolution_clock::now();
    nvtxRangePop(); 

    std::cout << "Kernel: " << std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count() << " ms" << std::endl;
    std::cout << "CUB:   " << std::chrono::duration<double, std::milli>(end_cub - start_cub).count() << " ms" << std::endl;

    nvtxRangePush("LUT Generation");
    
    thrust::inclusive_scan(thrust::device, d_hist.begin(), d_hist.end(), d_cdf.begin());

    thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> result_pair;
    result_pair = thrust::minmax_element(thrust::device, d_cdf.begin(), d_cdf.end());
    
    int cdf_min = *result_pair.first;
    float range = (float)(pixels - cdf_min);

    Normalize norm(cdf_min, range);
    auto transform_iter = thrust::make_transform_iterator(d_cdf.begin(), norm);
    thrust::copy(transform_iter, transform_iter + 256, d_lut.begin());

    nvtxRangePop();

    nvtxRangePush("Apply LUT");

    lut_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_in.data()),
        thrust::raw_pointer_cast(d_out.data()),
        thrust::raw_pointer_cast(d_lut.data()),
        pixels
    );
    cudaDeviceSynchronize();
    nvtxRangePop();

    cv::Mat result(img.rows, img.cols, CV_8U);
    thrust::copy(d_out.begin(), d_out.end(), result.data);

    cv::Mat ref;
    auto start_cv = std::chrono::high_resolution_clock::now();
    cv::equalizeHist(img, ref);
    auto end_cv = std::chrono::high_resolution_clock::now();
    std::cout << "opencv CPU:    " << std::chrono::duration<double, std::milli>(end_cv - start_cv).count() << " ms" << std::endl;

    cv::imwrite("wynik_gpu.png", result);
    cv::imwrite("wynik_opencv.png", ref);

    nvtxRangePop();
    return 0;
}