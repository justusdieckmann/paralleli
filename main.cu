#include <iostream>
#include "paralleli.h"
#include "mds.h"

using namespace pll;

class MapTestOp : public Operator<long, long> {
public:
    PLL_USERFUNC long operator()(long x) override {
        return 5;
    }
};

class IndexTestOp : public Operator<long, long, array<unsigned int, 3>&> {
public:
    PLL_USERFUNC long operator()(long x, array<unsigned int, 3> &coords) override {
        return coords[0] + 10 * coords[1] + 100 * coords[2];
    }
};

class MoveRightOp : public Operator<long, const PLS<long, 3>&, array<int, 3>> {
public:
    PLL_USERFUNC long operator()(const PLS<long, 3> &pls, array<int, 3> coords) override {
        return pls({coords[0] - 1, coords[1], coords[2]});
    }
};

int main() {
    Paralleli::init();

    IndexTestOp op;
    MoveRightOp stencilTestOp;
    MapTestOp mapTestOp;

    MDS<long, 3> mds({3, 3, 3});
    MDS<long, 3> mds2({3, 3, 3});

    mds.mapWithIndex(op, mds);
    mds.print();
    mds.mapStencil(stencilTestOp, mds2, 1, 0);
    mds2.print();

    return 0;
}
