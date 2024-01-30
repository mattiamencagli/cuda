#include<stdio.h>
#include<iostream>

#include <complex>
#include <vector>

#include <cuda_runtime.h>
#include <cufftXt.h>

#include "cufft_utils.h"

#define PRINTS

int main(int argc,char **argv) {

    std::cout << " ### EXAMPLE CUFFT ### " << std::endl;
    std::cout << " Check 1d_r2c in https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuFFT/ " << std::endl;
    std::cout << " See documentation https://docs.nvidia.com/cuda/cufft/index.html#cufft-code-examples " << std::endl;

    int n = 10000;
	int print_lim = 8;
    int batch_size = 2;
    int fft_size = batch_size * n;

    // example: real input, complex output
    std::vector<float> input(fft_size, 0);
    std::vector<std::complex<float>> output(static_cast<int>((fft_size * 0.5 + 1)));

    for (int i = 0; i < fft_size; i++)
        input[i] = static_cast<float>(i);

    std::cout << "Input array:" << std::endl;
    for (int i=0; i<print_lim; ++i) {
        std::cout << " " << input[i] << std::endl;
    }
    std::cout << std::endl;

    // Set cufft
    cufftHandle plan;
    CUFFT_CALL(cufftCreate(&plan));
    CUFFT_CALL(cufftPlan1d(&plan, input.size(), CUFFT_R2C, batch_size));

    // Create device stream
    cudaStream_t stream = NULL;
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUFFT_CALL(cufftSetStream(plan, stream));

    // Create device arrays
    float *d_input = nullptr;
    cufftComplex *d_output = nullptr;
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_input), sizeof(float) * input.size()));
    CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void **>(&d_output), sizeof(std::complex<float>) * output.size()));

    // copy input on gpu
    CUDA_RT_CALL(cudaMemcpyAsync(d_input, input.data(), sizeof(float) * input.size(), cudaMemcpyHostToDevice, stream));

    // cufft
    CUFFT_CALL(cufftExecR2C(plan, d_input, d_output));

    // copy ouput on cpu
    CUDA_RT_CALL(cudaMemcpyAsync(output.data(), d_output, sizeof(std::complex<float>) * output.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    std::cout << "Output array:" << std::endl;
    for (int i=0; i<print_lim; ++i) {
        std::cout << " " << output[i].real() <<"+" << output[i].imag() << "i" << std::endl;
    }
    std::cout << std::endl;

    // free resources 
    CUDA_RT_CALL(cudaFree(d_input));
    CUDA_RT_CALL(cudaFree(d_output));
    CUFFT_CALL(cufftDestroy(plan));
    CUDA_RT_CALL(cudaStreamDestroy(stream));
    CUDA_RT_CALL(cudaDeviceReset());

    std::cout << " ### END EXAMPLE CUFFT ### " << std::endl;

    return 0;
}
