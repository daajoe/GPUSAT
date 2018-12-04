#ifndef GPUSAT_GPUSAUTILS_H
#define GPUSAT_GPUSAUTILS_H
#if defined __CYGWIN__ || defined __MINGW32__
#define alloca __builtin_alloca
#endif

#include <CL/cl_platform.h>
#include <queue>
#include <chrono>
#include <numeric>

namespace gpusat {
    /**
     *
     * @param id
     * @param tree
     * @param numVars
     * @return
     */
    inline cl_double getCount(cl_long id, cl_long *tree, cl_int numVars) {
        int nextId = 0;
        for (int i = 0; i < numVars; i++) {
            nextId = ((int *) &(tree[nextId]))[(id >> (numVars - i - 1)) & 1];
            if (nextId == 0) {
                return 0.0;
            }
        }
        return *reinterpret_cast <cl_double *>(&tree[nextId]);
    }

    /**
     *
     * @return
     */
    inline long long int getTime() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }
}
#endif //GPUSAT_GPUSAUTILS_H
