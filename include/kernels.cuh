#ifndef PARALLELI_KERNELS_CUH
#define PARALLELI_KERNELS_CUH

#include <cuda.h>
#include "paralleli.h"
#include "array.h"
#include "pls.h"

namespace pll::kernel {

    template <unsigned int N>
    PLL_USERFUNC void indexToCoordinate(size_t index, array<unsigned int, N> globalSize, array<int, N> &coordinates) {
        for (size_t i = 0; i < N; i++) {
            coordinates[i] = index % globalSize[i];
            index /= globalSize[i];
        }
    }

    template <unsigned int N>
    PLL_USERFUNC void indexToCoordinate(size_t index, array<unsigned int, N> globalSize, array<unsigned int, N> &coordinates) {
        for (size_t i = 0; i < N; i++) {
            coordinates[i] = index % globalSize[i];
            index /= globalSize[i];
        }
    }

    template <typename R, typename T, typename F, typename = std::enable_if_t<std::is_base_of_v<Operator<R, T>, F>>>
    __global__ void map(size_t elements, R* out, T* in, F op) {
        size_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= elements) {
            return;
        }
        out[i] = op(in[i]);
    }

    template <typename R, typename T, unsigned int N, typename F, typename = std::enable_if_t<std::is_base_of_v<Operator<R, T, array<unsigned int, N>&>, F>>>
    __global__ void mapIndex(size_t start, size_t elements, R* out, T* in, F op, array<unsigned int, N> globalSize) {
        size_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= elements) {
            return;
        }
        array<unsigned int, N> coordinates;
        indexToCoordinate<N>(i + start, globalSize, coordinates);
        out[i] = op(in[i], coordinates);
    }

    template <typename R, typename T, unsigned int N, typename F, typename = std::enable_if_t<std::is_base_of_v<Operator<R, const PLS<T, N>&, array<int, N>>, F>>>
    __global__ void mapStencil(size_t start, size_t elements, R* out, F op, array<unsigned int, N> globalSize, PLS<T, N> pls) {
        size_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= elements) {
            return;
        }
        array<int, N> coordinates;
        indexToCoordinate<N>(i + start, globalSize, coordinates);
        out[i] = op(pls, coordinates);
    }

}

#endif //PARALLELI_KERNELS_CUH
