#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <iostream>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        dim3 threadsPerBlock(BLOCK_SIZE);

        __global__ void naiveScan(int n, int* odata, const int* idata, int off) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) return;

            if (index - off >= 0) {
                odata[index] = idata[index] + idata[index - off];
            } else {
                odata[index] = idata[index];
            }
        }

        __global__ void inclusiveToExclusiveScan(int n, const int* idata, int* scan) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) return;
            scan[index] -= idata[index];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* a, * b;
            cudaMalloc((void**)&a, n * sizeof(int));
            cudaMalloc((void**)&b, n * sizeof(int));
            cudaMemcpy(a, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            dim3 blockCount = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            int* src, *dst;
            int count = ilog2ceil(n);
            for (int i = 0; i < count; ++i) {
                src = i % 2 == 0 ? a : b;
                dst = i % 2 == 0 ? b : a;
                naiveScan << <blockCount, threadsPerBlock >> > (n, dst, src, 1 << i);
                if(i < count-1) cudaDeviceSynchronize();
            }

            cudaMemcpy(src, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            inclusiveToExclusiveScan << <blockCount, threadsPerBlock >> > (n, src, dst);
            cudaDeviceSynchronize();

            timer().endGpuTimer();

            cudaMemcpy(odata, dst, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(a);
            cudaFree(b);
        }
    }
}
