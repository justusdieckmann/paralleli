#include <iostream>
#include "paralleli.h"
#include "mds.h"

using namespace pll;

class InPlaceTestOp : public Operator<long, long, array<unsigned int, 3>&> {
public:
    PLL_USERFUNC long operator()(long x, array<unsigned int, 3> &coords) override {
        return coords[0] + 10 * coords[1] + 100 * coords[2];
    }
};

int main() {
    Paralleli::init();

    MDS<long, 3> mds({3, 3, 3});

    InPlaceTestOp op;

    mds.mapWithIndex(op, mds);

    mds.print();

    return 0;
}
