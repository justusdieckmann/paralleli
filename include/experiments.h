#ifndef PARALLELI_EXPERIMENTS_H
#define PARALLELI_EXPERIMENTS_H

#include "mds.h"

using namespace pll;

template <AccessType A, typename T, unsigned int N>
struct Wrapper {
    using type = T;
};

template <typename T, unsigned int N>
struct Wrapper<AccessType::field, T, N> {
    using type = T;
};

template <typename T, unsigned int N>
struct Wrapper<AccessType::everything, T, N> {
    using type = pll::MDS<T, N>;
};

#endif //PARALLELI_EXPERIMENTS_H
