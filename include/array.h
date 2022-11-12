#ifndef PARALLELI_ARRAY_H
#define PARALLELI_ARRAY_H

namespace pll {

    template<typename T, unsigned int N>
    struct array {
        T data[N];

        PLL_USERFUNC T& operator[](size_t n) {
            return data[n];
        }
    };

}

#endif //PARALLELI_ARRAY_H
