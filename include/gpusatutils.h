#ifndef GPUSAT_GPUSAUTILS_H
#define GPUSAT_GPUSAUTILS_H
#if defined __CYGWIN__ || defined __MINGW32__
#define alloca __builtin_alloca
#endif

#include <CL/cl_platform.h>
#include <queue>
#include <chrono>
#include <numeric>
#include <types.h>

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
    extern double getCount(long id, const TreeNode *tree, long numVars);

    /**
     * @return the time in millisecons since the epoch
     */
    inline long long int getTime() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }
}
#endif //GPUSAT_GPUSAUTILS_H
