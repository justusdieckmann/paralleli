//
// Created by justus on 11.11.22.
//

#ifndef PARALLELI_PARALLELI_H
#define PARALLELI_PARALLELI_H

#define PLL_USERFUNC __host__ __device__

#include <vector>
#include "cuda.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = false)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


namespace pll {

    namespace Paralleli {
        static unsigned int nGPUs;
        static unsigned int threadsPerBlock;
        static cudaStream_t *streams;

        static void init() {
            nGPUs = 2;
            threadsPerBlock = 512;
            streams = static_cast<cudaStream_t *>(malloc(sizeof(cudaStream_t) * nGPUs));
            for (int i = 0; i < nGPUs; i++) {
                cudaStreamCreate(&streams[i]);
            }
        }

        static void syncStreams() {
            for (int i = 0; i < nGPUs; i++) {
                cudaStreamSynchronize(streams[i]);
            }
        }
    }

    template <typename R, typename ...Ts>
    class Operator {
    public:
        PLL_USERFUNC virtual R operator()(Ts...) = 0;
        virtual ~Operator() = default;
    };

}

#endif //PARALLELI_PARALLELI_H
