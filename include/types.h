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
#include <assert.h>
#include <memory>
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

        ArraySolution() = default;
        ArraySolution(const ArraySolution&) = delete;
        ArraySolution& operator=(ArraySolution& other) = delete;

        // move constructor to ovoid copying solutions
        ArraySolution(ArraySolution&& other) = default;
        // move assignment
        ArraySolution& operator=(ArraySolution&& other) = default;
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

        TreeSolution() = default;
        TreeSolution(const TreeSolution&) = delete;
        TreeSolution& operator=(TreeSolution& other) = delete;

        // move constructor to ovoid copying solutions
        TreeSolution(TreeSolution&& other) = default;
        // move assignment
        TreeSolution& operator=(TreeSolution&& other) = default;
    };

    static_assert(sizeof(TreeNode) == sizeof(int64_t));

    struct GPUVars {
        /// Number of variables.
        int64_t count;
        /// Pointer to GPU memory containing the variable buffer.
        int64_t* vars;
    };

    template <class T>
    GPU_HOST_ATTR T dataStructureVisit(
        const std::variant<TreeSolution, ArraySolution>& solution,
        T(*treeFunc)(const TreeSolution& sol),
        T(*arrayFunc)(const ArraySolution& sol)
    ) {
        if (std::holds_alternative<TreeSolution>(solution)) {
            return treeFunc(std::get<TreeSolution>(solution));
        } else if (std::holds_alternative<ArraySolution>(solution)) {
            return arrayFunc(std::get<ArraySolution>(solution));
        }
        assert(0);
    }

    GPU_HOST_ATTR inline int64_t minId(const std::variant<TreeSolution, ArraySolution>& solution) {
        return dataStructureVisit<int64_t>(solution,
            [](const TreeSolution& sol) { return sol.minId; },
            [](const ArraySolution& sol) { return sol.minId; }
        );
    }

    GPU_HOST_ATTR inline int64_t maxId(const std::variant<TreeSolution, ArraySolution>& solution) {
        return dataStructureVisit<int64_t>(solution,
            [](const TreeSolution& sol) { return sol.maxId; },
            [](const ArraySolution& sol) { return sol.maxId; }
        );
    }

    GPU_HOST_ATTR inline size_t dataStructureSize(const std::variant<TreeSolution, ArraySolution>& solution) {
        return dataStructureVisit<size_t>(solution,
            [](const TreeSolution& sol) { return sol.size; },
            [](const ArraySolution& sol) { return sol.size; }
        );
    }

    GPU_HOST_ATTR inline int64_t numSolutions(const std::variant<TreeSolution, ArraySolution>& solution) {
        return dataStructureVisit<int64_t>(solution,
            [](const TreeSolution& sol) { return sol.numSolutions; },
            [](const ArraySolution& sol) { return sol.numSolutions; }
        );
    }

    inline int64_t* numSolutionsPtr(std::variant<TreeSolution, ArraySolution>& solution) {
        if (auto sol = std::get_if<TreeSolution>(&solution)) {
            return (int64_t*)&(sol->numSolutions);
        } else if (auto sol = std::get_if<ArraySolution>(&solution)) {
            return (int64_t*)&(sol->numSolutions);
        }
        return NULL;
    }

    GPU_HOST_ATTR inline int64_t* dataPtr(std::variant<TreeSolution, ArraySolution>& solution) {
        if (auto sol = std::get_if<TreeSolution>(&solution)) {
            return (int64_t*)sol->tree;
        } else if (auto sol = std::get_if<ArraySolution>(&solution)) {
            return (int64_t*)sol->elements;
        }
        return NULL;
    }

    GPU_HOST_ATTR inline bool dataEmpty(const std::variant<TreeSolution, ArraySolution>& solution) {
        return dataStructureVisit<bool>(solution,
            [](const TreeSolution& sol) { return sol.tree == NULL; },
            [](const ArraySolution& sol) { return sol.elements == NULL; }
        );
    }

    inline void freeData(std::variant<TreeSolution, ArraySolution>& solution) {
        if (auto sol = std::get_if<TreeSolution>(&solution)) {
            if (sol->tree != NULL) {
                delete[] sol->tree;
                sol->tree = NULL;
            }
        } else if (auto sol = std::get_if<ArraySolution>(&solution)) {
            if (sol->elements != NULL) {
                delete[] sol->elements;
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
        std::vector<BagType> edges;
        std::vector<std::variant<TreeSolution, ArraySolution>> solution;
        int64_t maxSize = 0;

        BagType() = default;

        BagType(const BagType&) = delete;
        BagType& operator=(BagType& other) = delete;

        // move constructor to ovoid copying solutions
        BagType(BagType&& other) = default;
        // move assignment
        BagType& operator=(BagType&& other) = default;
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
    inline bool compTreedType(const BagType& a, const BagType& b) {
        return a.id < b.id;
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
