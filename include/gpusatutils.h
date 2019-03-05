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
     * returns the model count which corresponds to the given id
     *
     * @param id
     *      the id for which the model count should be returned
     * @param tree
     *      a pointer to the tree structure
     * @param numVars
     *      the number of variables in the bag
     * @return
     *      the model count
     */
    inline cl_double getCount(cl_long id, cl_long *tree, cl_long numVars) {
        cl_long nextId = 0;
        for (cl_long i = 0; i < numVars; i++) {
            nextId = ((cl_uint *) &(tree[nextId]))[(id >> (numVars - i - 1)) & 1];
            if (nextId == 0) {
                return 0.0;
            }
        }
        return *reinterpret_cast <cl_double *>(&tree[nextId]);
    }

    /**
     * @return the time in millisecons since the epoch
     */
    inline long long int getTime() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }
}
#endif //GPUSAT_GPUSAUTILS_H
