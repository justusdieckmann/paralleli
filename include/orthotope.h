/**
 * Copyright (c) 2022 Justus Dieckmann
 *
 * This software is released under the MIT License.
 * https://opensource.org/licenses/MIT
 */

#ifndef PARALLELI_ORTHOTOPE_H
#define PARALLELI_ORTHOTOPE_H

#include "array.h"

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
    template<unsigned int N, typename T>
    class Orthotope {
    public:
        Orthotope() : min({}), max({}) {
        }

        Orthotope(array<T, N> min, array<T, N> max) : min(min), max(max) {
        }

        [[nodiscard]] constexpr unsigned int dim() const{
            return N;
        }

        bool isInside(array<T, N> point) {
            for (unsigned int i = 0; i < N; i++) {
                if (point[i] < min[i] || point[i] >= max[i]) {
                    return false;
                }
            }
            return true;
        }

        T size() {
            T volume = 1;
            for (unsigned int i = 0; i < N; i++) {
                volume *= (max[i] - min[i]);
            }
            return volume;
        }

        T getDimensionLength(unsigned int dim) {
            return max[dim] - min[dim];
        }

        int getDimensionMin(unsigned int dim) {
            return min[dim];
        }

        int getDimensionMax(unsigned int dim){
            return max[dim];
        }

        const array<T, N> min;
        const array<T, N> max;
    };

}

#endif //PARALLELI_ORTHOTOPE_H