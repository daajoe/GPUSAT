#include <types.h>

namespace gpusat {
    bool compTreedType(const preebagType *a, const preebagType *b) {
        return a->id < b->id;
    }
}