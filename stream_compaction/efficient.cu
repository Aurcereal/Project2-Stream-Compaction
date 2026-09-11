#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

#define DEBUG_MESSAGES 0

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        dim3 threadsPerBlock(BLOCK_SIZE);

        // Split array into uniform segments, add half point of each segment to its end point
        __global__ void upSweep(int count, int* data, int halfSegment) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= count) return;

            data[index * 2 * halfSegment + 2 * halfSegment - 1] += data[index * 2 * halfSegment + halfSegment - 1];
        }

        __global__ void downSweep(int count, int* data, int siblingOffset) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= count) return;

            int leftIndex = index * 2 * siblingOffset + siblingOffset-1;
            int rightIndex = index * 2 * siblingOffset + 2 * siblingOffset - 1;

            int parentValue = data[rightIndex];
            int leftValue = data[leftIndex];

            data[leftIndex] = parentValue;
            data[rightIndex] = parentValue + leftValue;

        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
#if DEBUG_MESSAGES
            {
                std::cout << "Start Data: ";
                for (int i = 0; i < n; i++) {
                    std::cout << idata[i] << ", ";
                }
                std::cout << std::endl;
            }
#endif

            int p2 = ilog2ceil(n);
            int p2n = 1 << p2;
            int* a;
            cudaMalloc((void**)&a, p2n * sizeof(int));
            cudaMemset(a, 0, p2n * sizeof(int));
            cudaMemcpy(a, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // Up Sweep
            int* upSweepData = a;
            for (int i = 0; i < p2; ++i) {
                int count = p2n >> (i+1);
                int halfSegment = 1 << i;
#if DEBUG_MESSAGES
                std::cout << "i: " << i << " count: " << count << " halfSegment: " << halfSegment << std::endl;
#endif
                dim3 blockCount = dim3((count + BLOCK_SIZE - 1) / BLOCK_SIZE);
                upSweep << <blockCount, threadsPerBlock >> > (count, upSweepData, halfSegment);

                cudaDeviceSynchronize();
            }

#if DEBUG_MESSAGES
            {
                int* chk = new int[n];
                cudaMemcpy(chk, a, n * sizeof(int), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                std::cout << "Up Sweep Result: ";
                for (int i = 0; i < n; i++) {
                    std::cout << chk[i] << ", ";
                }
                std::cout << std::endl;
                delete[] chk;
            }
#endif

            // Set Root to 0
            int* aLastElem = a + (p2n - 1);
            cudaMemset(aLastElem, 0, 1 * sizeof(int));

#if DEBUG_MESSAGES
            {
                int* chk = new int[p2n];
                cudaMemcpy(chk, a, p2n * sizeof(int), cudaMemcpyDeviceToHost);
                if (chk[p2n - 1] != 0) std::cout << " ERROR !!!" << std::endl;
                delete[] chk;
            }
#endif

            // Down Sweep
            int* downSweepData = a;
            for (int i = 0; i < p2; i++) {
                int count = 1 << i;
                int halfSegment = p2n >> (i+1);
                dim3 blockCount = dim3((count + BLOCK_SIZE - 1) / BLOCK_SIZE);
                downSweep << <blockCount, threadsPerBlock >> > (count, downSweepData, halfSegment);

                cudaDeviceSynchronize();

#if DEBUG_MESSAGES
                std::cout << "down i: " << i << " count: " << count << " halfSegment: " << halfSegment << std::endl;
                
                {
                    int* chk = new int[10];
                    cudaMemcpy(chk, upSweepData, 10 * sizeof(int), cudaMemcpyDeviceToHost);
                    std::cout << "Down sweep after i: " << i << ": ";
                    for (int i = 0; i < 10; i++) {
                        std::cout << chk[i] << ", ";
                    }
                    std::cout << std::endl;
                    delete[] chk;
                }

#endif
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, downSweepData, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(a);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) { // Way more transfers btwn CPU & GPU than necessary since we only have scan CPU func
            int* alive, * aliveExclusiveSum, *idataDevice, *odataDevice;
            cudaMalloc((void**)&alive, n * sizeof(int));
            cudaMalloc((void**)&aliveExclusiveSum, n * sizeof(int));
            cudaMalloc((void**)&idataDevice, n * sizeof(int));
            cudaMalloc((void**)&odataDevice, n * sizeof(int));

            cudaMemcpy(idataDevice, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            timer().setGpuTimerLock(true);

            dim3 blockCount = dim3((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            Common::kernMapToBoolean << <blockCount, threadsPerBlock >> > (n, alive, idataDevice);
            cudaDeviceSynchronize();
            int* aliveCpu = new int[n];
            cudaMemcpy(aliveCpu, alive, n * sizeof(int), cudaMemcpyDeviceToHost);

            int* aliveExclusiveSumCpu = new int[n];
            scan(n, aliveExclusiveSumCpu, aliveCpu);
            int count = aliveExclusiveSumCpu[n - 1] + aliveCpu[n - 1];
            cudaMemcpy(aliveExclusiveSum, aliveExclusiveSumCpu, n * sizeof(int), cudaMemcpyHostToDevice);

            Common::kernScatter << <blockCount, threadsPerBlock >> > (n, odataDevice, idataDevice, alive, aliveExclusiveSum);
            cudaDeviceSynchronize();

            timer().setGpuTimerLock(false);
            timer().endGpuTimer();

            cudaMemcpy(odata, odataDevice, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(alive);
            cudaFree(aliveExclusiveSum);
            cudaFree(idataDevice);
            cudaFree(odataDevice);
            delete[] aliveCpu;
            delete[] aliveExclusiveSumCpu;

            return count;
        }
    }
}
