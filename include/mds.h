/**
 * Copyright (c) 2022 Justus Dieckmann
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */

#ifndef PARALLELI_MDS_H
#define PARALLELI_MDS_H

#include "paralleli.h"
#include "orthotope.h"
#include "memory_section.h"
#include "kernels.cuh"
#include "pls.h"
#include <cstdlib>
#include <cuda.h>
#include <vector>

namespace pll {

    std::ostream& operator<<(std::ostream& os, const float4 f) {
        os << '(' << f.x << ", " << f.y << ", " << f.z << ", " << f.w << ')';
        return os;
    }

    /**
     * \brief Class MDS represents a distributed structure.
     *
     * \tparam T Element type. Restricted to classes without pointer data members.
     * \tparam N Dimensions.
     */
    template<typename T, unsigned int N>
    class MDS {
    public:

        MDS() = default;

        /**
         * \brief Creates an empty distributed array.
         *
         * @param size Size of the distributed array.
         */
        explicit MDS(array<unsigned int, N> size) :
                globalSize(size),
                localSection({}, size) {
            this->init();
        }

        /**
         * \brief Destructor.
         */
        ~MDS() = default;

        [[nodiscard]] constexpr size_t dim() const {
            return N;
        }

        void fill(T element) {
            for (size_t i = 0; i < localSection.size(); i++) {
                localPartition[i] = element;
            }
            markHostAhead();
        }

        template <typename R, typename F, typename = std::enable_if_t<std::is_base_of_v<Operator<R, T>, F>>>
        void map(F& op, MDS<R, N> &result) {
            ensureDeviceUpToDate();
            for (size_t i = 0; i < Paralleli::nGPUs; i++) {
                auto memoryRegion = deviceMemoryRegions[i];
                cudaSetDevice(memoryRegion.device);
                size_t threadsPerBlock = Paralleli::threadsPerBlock;
                size_t numBlocks = (memoryRegion.getSize() + threadsPerBlock - 1) / threadsPerBlock;
                kernel::map<<<numBlocks, threadsPerBlock, 0, Paralleli::streams[i]>>>(
                        localSection.size(), result.deviceMemoryRegions[i].data, memoryRegion.data, op
                );
                memoryRegion.isAhead = true;
            }
            Paralleli::syncStreams();
        }

        template <typename R, typename F, typename = std::enable_if_t<std::is_base_of_v<Operator<T, T, array<unsigned int, N>&>, F>>>
        void mapWithIndex(F& op, MDS<R, N> &result) {
            ensureDeviceUpToDate();
            for (size_t i = 0; i < Paralleli::nGPUs; i++) {
                auto& memoryRegion = deviceMemoryRegions[i];
                cudaSetDevice(memoryRegion.device);
                size_t threadsPerBlock = Paralleli::threadsPerBlock;
                size_t numBlocks = (memoryRegion.getSize() + threadsPerBlock - 1) / threadsPerBlock;
                kernel::mapIndex<T, T, N, F><<<numBlocks, threadsPerBlock, 0, Paralleli::streams[i]>>>(
                        localSection.size(), result.deviceMemoryRegions[i].data, memoryRegion.data, op, globalSize
                );
                memoryRegion.isAhead = true;
            }
        }

        template <typename R, typename F, typename = std::enable_if_t<std::is_base_of_v<Operator<R, const PLS<T, N>&, array<int, N>>, F>>>
        void mapStencil(F& op, MDS<R, N> &result, size_t stencilSize, T neutralValue) {
            ensureDeviceUpToDate();
            syncPLS(stencilSize, neutralValue);
            for (size_t i = 0; i < Paralleli::nGPUs; i++) {
                auto memoryRegion = deviceMemoryRegions[i];
                cudaSetDevice(memoryRegion.device);
                size_t threadsPerBlock = Paralleli::threadsPerBlock;
                size_t numBlocks = (memoryRegion.getSize() + threadsPerBlock - 1) / threadsPerBlock;
                kernel::mapStencil<R, T, N, F><<<numBlocks, threadsPerBlock, 0, Paralleli::streams[i]>>>(
                        localSection.size(), result.deviceMemoryRegions[i].data, op, globalSize, pls[i]
                );
                result.deviceMemoryRegions[i].isAhead = true;
            }
            Paralleli::syncStreams();
        }

        void ensureHostUpToDate() {
            for (GPUMemorySection<T> &memoryRegion : deviceMemoryRegions) {
                if (memoryRegion.isAhead) {
                    cudaSetDevice(memoryRegion.device);
                    cudaMemcpyAsync(memoryRegion.hostData, memoryRegion.data, memoryRegion.getBytes(),
                                    cudaMemcpyDeviceToHost, Paralleli::streams[memoryRegion.device]);
                    memoryRegion.isAhead = false;
                }
            }
            Paralleli::syncStreams();
        }

        void ensureDeviceUpToDate() {
            for (GPUMemorySection<T> &memoryRegion : deviceMemoryRegions) {
                if (memoryRegion.isBehind) {
                    cudaSetDevice(memoryRegion.device);
                    cudaMemcpyAsync(memoryRegion.data, memoryRegion.hostData, memoryRegion.getBytes(),
                                    cudaMemcpyHostToDevice, Paralleli::streams[memoryRegion.device]);
                    memoryRegion.isBehind = false;
                }
            }
            Paralleli::syncStreams();
        }

        void markDeviceAhead() {
            for (GPUMemorySection<T> &memoryRegion : deviceMemoryRegions) {
                memoryRegion.isAhead = true;
            }
        }

        void markHostAhead() {
            for (GPUMemorySection<T> &memoryRegion : deviceMemoryRegions) {
                memoryRegion.isBehind = true;
            }
        }

        void print() {
            ensureHostUpToDate();

            for (int i = 0; i < localSection.size(); i++) {
                std::cout << localPartition[i] << ", ";
            }
            std::cout << std::endl;
        }

        T *getLocalPartition() {
            return localPartition;
        }

    private:
        void init() {
            localPartition = new T[localSection.size()];
            calculatePartition();
        }

        void initPLS(int stencilSize, T neutralValue) {
            pls = std::vector<PLS<T, N>>();
            pls.reserve(Paralleli::nGPUs);
            for (int i = 0; i < Paralleli::nGPUs; i++) {
                pls.push_back(PLS<T, N>(
                        globalSize,
                        *(array<int, N> *) &localSection.min,
                        *(array<int, N> *) &localSection.max,
                        stencilSize,
                        neutralValue,
                        this->deviceMemoryRegions[i].data
                ));
            }
            supportedStencilSize = stencilSize;
        }

        void syncPLS(int stencilSize, T neutralValue) {
            if (stencilSize > supportedStencilSize) {
                initPLS(stencilSize, neutralValue);
            }
            for(int i = 1; i < Paralleli::nGPUs; i++) {
                size_t bottomPaddingSize = pls[i - 1].getBottomPaddingElements() * sizeof(T);
                cudaMemcpyPeerAsync(
                        pls[i - 1].bottomPadding, i - 1, pls[i].data, i,
                        bottomPaddingSize, Paralleli::streams[i - 1]
                );

                size_t topPaddingSize = pls[i].getTopPaddingElements() * sizeof(T);
                cudaMemcpyPeerAsync(
                        pls[i].topPadding, i, pls[i - 1].data + (deviceMemoryRegions[i].getBytes() - topPaddingSize), i - 1,
                        topPaddingSize, Paralleli::streams[i - 1]
                );
            }
            Paralleli::syncStreams();
        }

        void calculatePartition() {
            size_t ngpu = pll::Paralleli::nGPUs;

            int length = localSection.getDimensionLength(N - 1);

            deviceMemoryRegions = std::vector<GPUMemorySection<T>>(Paralleli::nGPUs);

            layerSize = 1;
            for (int i = 0; i < N - 1; i++) {
                layerSize *= localSection.getDimensionLength(i);
            }

            size_t lowerSize = length / ngpu;
            size_t remaining = length % ngpu;
            size_t current = 0;

            for (int i = 0; i < ngpu; i++) {
                size_t min = current;
                current += lowerSize;
                if (i < remaining) {
                    current++;
                }
                deviceMemoryRegions[i] = GPUMemorySection<T>(min, current - 1, layerSize, i, &localPartition[min]);
            }
        }

    private:
        array<unsigned int, N> globalSize;
        size_t layerSize{};
        Orthotope<N, unsigned int> localSection;
        // local partition
        T *localPartition;
        std::vector<PLS<T, N>> pls;
        std::vector<GPUMemorySection<T>> deviceMemoryRegions;
        int supportedStencilSize = 0;
    };

}

#endif //PARALLELI_MDS_H