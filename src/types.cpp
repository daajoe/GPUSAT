#include <types.h>

namespace gpusat {
    bool compTreedType(const preebagType *a, const preebagType *b) {
        return a->id < b->id;
    }

    bool compVars(const cl_long &a, const cl_long &b) {
        return std::abs(a) < std::abs(b);
    }
}