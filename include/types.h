#ifndef GPUSAT_TYPES_H_H
#define GPUSAT_TYPES_H_H
#if defined __CYGWIN__ || defined __MINGW32__
#define alloca __builtin_alloca
#endif

#include <CL/cl_platform.h>
#include <cmath>
#include <list>
#include <vector>
#include <variant>
#include <set>
#include <stdint.h>

namespace gpusat {

    /// array type for storing the models
    struct ArraySolution {
        double *elements = nullptr;
        int64_t numSolutions = 0;
        size_t size = 0;
        int64_t minId = 0;
        int64_t maxId = 0;
    };

    union TreeNode {
        int64_t empty = 0;
        struct {
            uint32_t lowerIdx;
            uint32_t upperIdx;
        } __attribute__((packed));
        double content;
    } __attribute__((packed));
    /// tree type for storing the models
    struct TreeSolution {
        TreeNode *tree = nullptr;
        int64_t numSolutions = 0;
        size_t size = 0;
        int64_t minId = 0;
        int64_t maxId = 0;
    };

    static_assert(sizeof(TreeNode) == sizeof(int64_t));

    struct GPUVars {
        /// Number of variables.
        int64_t count;
        /// Pointer to GPU memory containing the variable buffer.
        int64_t* vars;
    };


    inline int64_t minId(std::variant<TreeSolution, ArraySolution>* solution) {
        return std::visit([](auto sol) -> int64_t { return sol.minId; }, *solution);
    }

    inline int64_t maxId(std::variant<TreeSolution, ArraySolution>* solution) {
        return std::visit([](auto sol) -> int64_t { return sol.maxId; }, *solution);
    }

    inline size_t dataStructureSize(std::variant<TreeSolution, ArraySolution>* solution) {
        return std::visit([](auto sol) -> int64_t { return sol.size; }, *solution);
    }

    inline int64_t numSolutions(std::variant<TreeSolution, ArraySolution>* solution) {
        return std::visit([](auto sol) -> int64_t { return sol.numSolutions; }, *solution);
    }

    inline int64_t* numSolutionsPtr(std::variant<TreeSolution, ArraySolution>* solution) {
        if (auto sol = std::get_if<TreeSolution>(solution)) {
            return (int64_t*)&(sol->numSolutions);
        } else if (auto sol = std::get_if<ArraySolution>(solution)) {
            return (int64_t*)&(sol->numSolutions);
        }
        return NULL;
    }

    inline int64_t* dataPtr(std::variant<TreeSolution, ArraySolution>* solution) {
        if (auto sol = std::get_if<TreeSolution>(solution)) {
            return (int64_t*)sol->tree;
        } else if (auto sol = std::get_if<ArraySolution>(solution)) {
            return (int64_t*)sol->elements;
        }
        return NULL;
    }

    inline void freeData(std::variant<TreeSolution, ArraySolution>* solution) {
        if (auto sol = std::get_if<TreeSolution>(solution)) {
            if (sol->tree != NULL) {
                free(sol->tree);
                sol->tree = NULL;
            }
        } else if (auto sol = std::get_if<ArraySolution>(solution)) {
            if (sol->elements != NULL) {
                free(sol->elements);
                sol->elements = NULL;
            }
        }
    }

    /// type for a bag in the tree decomposition
    struct BagType {
        int64_t correction = 0;
        int64_t exponent = 0;
        int64_t id = 0;
        std::vector<int64_t> variables;
        std::vector<BagType *> edges;
        std::vector<std::variant<TreeSolution, ArraySolution>> solution;
        int64_t maxSize = 0;
    };

    /// type for saving a tree decomposition
    struct treedecType {
        int64_t numb = 0;
        int64_t numVars = 0;
        int64_t width = 0;
        std::vector<BagType> bags;
    };

    /**
     * Function that compares two tree decompostions by id.
     *
     * @param a
     *      the first tree decompostion
     * @param b
     *      the second tree decomposition
     * @return
     *      a < b
     */
    inline bool compTreedType(const BagType *a, const BagType *b) {
        return a->id < b->id;
    }

    /// type for saving the sat formula
    struct satformulaType {
        int64_t numVars = 0;
        int64_t numWeights = 0;
        bool unsat = false;
        double *variableWeights = nullptr;
        std::vector<std::vector<int64_t>> clauses;
        std::vector<int64_t> facts;
    };

    /// the graph type which was the base for the tree decomposition
    enum nodeType {
        JOIN, INTRODUCEFORGET
    };

    ///
    enum dataStructure {
        ARRAY, TREE
    };

    enum SolveMode {
        DEFAULT = 0,
        ARRAY_TYPE = 1 << 0,
        NO_EXP = 1 << 1,
    };

    GPU_HOST_ATTR inline SolveMode operator|(SolveMode a, SolveMode b)
    {
        return static_cast<SolveMode>(static_cast<int>(a) | static_cast<int>(b));
    }

    GPU_HOST_ATTR inline SolveMode operator&(SolveMode a, SolveMode b)
    {
        return static_cast<SolveMode>(static_cast<int>(a) & static_cast<int>(b));
    }

    /**
     * Function that compares two variables.
     *
     * @param a
     *      the first variable
     * @param b
     *      the second variable
     * @return
     *      true if abs a < b
     */
    inline bool compVars(const cl_long &a, const cl_long &b) {
        return std::abs(a) < std::abs(b);
    }
}

#endif //GPUSAT_TYPES_H_H
