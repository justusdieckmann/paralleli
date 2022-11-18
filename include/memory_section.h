/**
 * Copyright (c) 2022 Justus Dieckmann
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */

#ifndef PARALLELI_MEMORY_SECTION_H
#define PARALLELI_MEMORY_SECTION_H

#include "cuda.h"

namespace pll {
/**
 * \brief Class DA represents a distributed array.
 *
 * A distributed array represents a one-dimensional parallel container and is
 * distributed among all MPI processes the application was started with. It
 * includes data parallel skeletons such as map, mapStencil, zip, and fold as
 * well as variants of these skeletons.
 *
 * \tparam T Element type. Restricted to classes without pointer data members.
 */
    template <typename T>
    class GPUMemorySection {
    public:
        GPUMemorySection() = default;

        GPUMemorySection(size_t minLayer, size_t maxLayer, size_t layerSize, int device, T* hostData)
                : minLayer(minLayer),
                  maxLayer(maxLayer),
                  layerSize(layerSize),
                  device(device),
                  hostData(hostData){
            cudaSetDevice(device);
            cudaMalloc(&data, getBytes());
        }

        inline size_t getSize() {
            return layerSize * (maxLayer - minLayer + 1);
        }

        inline size_t getBytes() {
            return getSize() * sizeof(T);
        }

        inline size_t getFirstElement() {
            return minLayer * layerSize;
        }

        size_t minLayer;
        size_t maxLayer;
        size_t layerSize;
        int device;
        bool isAhead = false;
        bool isBehind = false;
        T* data;
        T* hostData;
    };
}

#endif //PARALLELI_MEMORY_SECTION_H