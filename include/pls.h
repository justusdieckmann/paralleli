#ifndef PARALLELI_PLS_H
#define PARALLELI_PLS_H

#include "paralleli.h"
#include "array.h"

namespace pll {

/**
 * \brief Class PLCube represents a padded local cube (partition). It serves
 *        as input for the mapStencil skeleton and actually is a shallow copy that
 *        only stores the pointers to the data. The data itself is managed by the
 *        mapStencil skeleton. For the user, the only important part is the \em get
 *        function.
 *
 * @tparam T The element type.
 */
    template <typename T, unsigned int N>
    class PLS {
    private:
        const array<unsigned int, N> globalSize;
        const array<int, 3> start;
        const array<int, 3> end;
    public:
        const int stencilSize;
        const int dataStartIndex = 0;
        const int dataEndIndex = 0;
        const int topPaddingStartIndex = 0;
        const int bottomPaddingEndIndex = 0;
        const T neutralValue;
        T *data;
        T *topPadding;
        T *bottomPadding;

        /**
         * \brief Constructor: creates a PLCube.
         */
        PLS(array<unsigned int, N> globalSize, array<int, N> start, array<int, N> end,
               int stencilSize, T neutralValue, T *data)
                : globalSize(globalSize), start(start), end(end), stencilSize(stencilSize), data(data), neutralValue(neutralValue),
                  dataStartIndex(coordinateToIndex(start)),
                  dataEndIndex(coordinateToIndex(end)),
                  topPaddingStartIndex(coordinateToIndex(one(start))),
                  bottomPaddingEndIndex(coordinateToIndex(two(end))) {
            cudaMalloc(&topPadding, (dataStartIndex - topPaddingStartIndex) * sizeof(T));
            cudaMalloc(&bottomPadding, (bottomPaddingEndIndex - dataEndIndex) * sizeof(T));
        }

        template <typename T2>
        array<T2, N> one(const array<T2, N> &a) {
            array<T2, N> arr;
            for (int i = 0; i < N; i++) {
                arr[i] = std::max(a[i] - stencilSize, 0);
            }
            return arr;
        }

        template <typename T2>
        array<T2, N> two(const array<T2, N> &a) {
            array<T2, N> arr;
            for (int i = 0; i < N; i++) {
                arr[i] = std::min(a[i], (int) globalSize[i]);
            }
            return arr;
        }

        PLL_USERFUNC T operator() (array<int, N> coords) const {
            for (int i = 0; i < N; i++) {
                if (coords[i] < 0 || coords[i] > globalSize[i]) {
                    return neutralValue;
                }
            }
            int index = coordinateToIndex(coords);
            if (index >= dataStartIndex) {
                if (index > dataEndIndex) {
                    return bottomPadding[index - dataEndIndex - 1];
                } else {
                    return data[index - dataStartIndex];
                }
            } else {
                return topPadding[index - topPaddingStartIndex];
            }
        }

        PLL_USERFUNC inline int coordinateToIndex(array<int, 3> coords) const {
            return coordinateToIndex<0>(coords);
        }

        template <unsigned int M>
        PLL_USERFUNC inline int coordinateToIndex(const array<int, N> &coords) const {
            return coordinateToIndex<M + 1>(coords) * globalSize[M] + coords[M];
        }

        template <>
        PLL_USERFUNC inline int coordinateToIndex<N - 1>(const array<int, N> &coords) const {
            return coords[N - 1];
        }

        PLL_USERFUNC array<unsigned int, N> indexToCoordinate(int index) const {
            array<unsigned int, N> coordinates;
            for (size_t i = 0; i < N; i++) {
                coordinates[i] = index % globalSize[i];
                index /= globalSize[i];
            }
            return coordinates;
        }

        inline int getTopPaddingElements() {
            return dataStartIndex - topPaddingStartIndex;
        }

        inline int getBottomPaddingElements() {
            return bottomPaddingEndIndex - dataEndIndex;
        }
    };
}

#endif //PARALLELI_PLS_H
