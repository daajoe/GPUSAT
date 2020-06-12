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
    inline double getCount(int64_t id, TreeNode *tree, int64_t numVars) {
        TreeNode current = tree[0];
        for (int64_t i = 0; i < numVars; i++) {
            if ((id >> (numVars - i - 1)) & 1) {
                current = tree[current.upperIdx];
            } else { 
                current = tree[current.lowerIdx];
            }
            if (current.empty == 0) {
                return 0.0;
            }
        }
        return current.content;
    }

    /**
     * @return the time in millisecons since the epoch
     */
    inline long long int getTime() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    }
}
#endif //GPUSAT_GPUSAUTILS_H
