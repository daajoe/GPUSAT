#pragma once
#ifndef GPUSAT_TYPES_H_H
#define GPUSAT_TYPES_H_H
#if defined __CYGWIN__ || defined __MINGW32__
#define alloca __builtin_alloca
#endif

#include <cmath>
#include <list>
#include <vector>
#include <variant>
#include <assert.h>
#include <memory>
#include <string.h>
#include <set>
#include <stdint.h>
#include <signal.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <atomic>

namespace gpusat {
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
    {
       if (code != cudaSuccess)
       {
          fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
          if (abort) raise(SIGABRT);
       }
       code = cudaGetLastError();
       if (code != cudaSuccess)
       {
          fprintf(stderr,"GPUassert last error: %s %s %d\n", cudaGetErrorString(code), file, line);
          if (abort) raise(SIGABRT);
       }
    }

    ///
    enum dataStructure {
        ARRAY, TREE
    };

    // taken from boost
    template <class T>
    inline void hash_combine(std::size_t& seed, const T& v)
    {
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    }

#if defined(__CUDACC__)
    extern __device__ uint64_t atomicAdd(uint64_t* address, uint64_t val);
    extern __device__ uint64_t atomicSub(uint64_t* address, uint64_t val);
#endif

    class Solution {
        public:
            /**
             * Pointer to the data of this container.
             */
            GPU_HOST_ATTR virtual void* data() const = 0;

            /**
             * Free the solution data of this container.
             */
            virtual void freeData() = 0;

            /**
             * Does this container still have solution data?
             */
            GPU_HOST_ATTR virtual bool hasData() const {
                return data() != nullptr;
            }

            /**
             * Minimal solution ID of this container.
             */
            GPU_HOST_ATTR virtual int64_t minId() const {
                return minId_;
            }

            /**
             * Maximal solution ID of this container.
             */
            GPU_HOST_ATTR virtual int64_t maxId() const {
                return maxId_;
            }

            /**
             * increase the solution counter by one.
             * FIXME: This is only used in ArraySolution,
             * but is generic because of type erasure in the kernel...
             */
            virtual __device__ void incSolutions() = 0;

            /**
             * Set minimal solution ID of this container.
             */
            virtual void setMinId(int64_t id) {
                minId_ = id;
            }

            /**
             * Set maximal solution ID of this container.
             */
            virtual void setMaxId(int64_t id) {
                maxId_ = id;
            }

            /**
             * Get the stored model count for some solution id.
             */
            GPU_HOST_ATTR virtual double solutionCountFor(int64_t id) const = 0;

            /**
             * Store the model count for some solution id.
             */
            __device__ virtual void setCount(int64_t id, double val) = 0;

            /**
             * Size of the container data structure in elements.
             */
            virtual size_t dataStructureSize() const {
                return size_;
            }

            /**
             * Size of one container element in bytes.
             */
            virtual size_t elementSize() const = 0;


            /**
             * A check sum of the solution container and its properties.
             * FIXME: still based on legacy data structures for comparison.
             */
            virtual size_t hash() const = 0;

            Solution(size_t size, int64_t minId, int64_t maxId) :
                size_(size), minId_(minId), maxId_(maxId) {};

            Solution(const Solution&) = delete;
            Solution& operator=(Solution& other) = delete;

            // move constructor to ovoid copying solutions
            Solution(Solution&& other) = default;
            // move assignment
            Solution& operator=(Solution&& other) = default;
        protected:
            size_t size_ = 0;
            int64_t minId_ = 0;
            int64_t maxId_ = 0;
    };

    class ArraySolution: public Solution {
        public:
            ArraySolution(size_t size, int64_t minId, int64_t maxId) :
                Solution(size, minId, maxId)
            {
                elements = (double*)malloc(size_ * elementSize());
                assert(elements != nullptr);
                memset(elements, 0, size_ * elementSize());
            }

            size_t elementSize() const {
                return sizeof(double);
            }

            void freeData() {
                free(elements);
                elements = nullptr;
            }

            ArraySolution* gpuCopyWithData(uint64_t* element_buffer) const {

                ArraySolution* gpu = nullptr;
                ArraySolution* cpu = (ArraySolution*)malloc(sizeof(ArraySolution));
                assert(cpu != nullptr);
                memcpy(cpu, this, sizeof(ArraySolution));
                cpu->elements = (double*)element_buffer;
                gpuErrchk(cudaMalloc(&gpu, elementSize() * dataStructureSize()));
                gpuErrchk(cudaMemcpy(gpu, cpu, sizeof(ArraySolution), cudaMemcpyDeviceToHost));
                free(cpu);
                return gpu;
            };

            static ArraySolution fromGPU(const ArraySolution* gpu) {
                ArraySolution* cpu = (ArraySolution*)malloc(sizeof(ArraySolution));
                assert(cpu != nullptr);
                gpuErrchk(cudaMemcpy(cpu, gpu, sizeof(ArraySolution), cudaMemcpyDeviceToHost));
                auto solution = ArraySolution(cpu->size_, cpu->minId(), cpu->maxId());
                solution.numSolutions = cpu->solutions();
                gpuErrchk(cudaMemcpy(solution.elements, cpu->elements, solution.elementSize() * solution.dataStructureSize(), cudaMemcpyDeviceToHost));
                return std::move(solution);
            }

            size_t hash() const {
                size_t h = dataStructureSize();
                hash_combine(h, minId());
                hash_combine(h, maxId());
                hash_combine(h, solutions());
                if (elements == NULL) {
                    return h;
                }
                for (int i=0; i < dataStructureSize(); i++) {
                    hash_combine(h, *reinterpret_cast<uint64_t*>(&elements[i]));
                }
                return h;
            }

            GPU_HOST_ATTR double solutionCountFor(int64_t id) const {
                return elements[id - minId_];
            }

            /**
             * increase the solution counter by one.
             */
            __device__ void incSolutions()  {
#if defined(__CUDACC__)
                atomicAdd(&numSolutions, 1);
#else
                fprintf(stderr, "this may not be called from CPU (yet)!");
                assert(false);
#endif
            }

            __device__ void decSolutions()  {
#if defined(__CUDACC__)
                atomicSub(&numSolutions, 1);
#else
                fprintf(stderr, "this may not be called from CPU (yet)!");
                assert(false);
#endif
            }

            /**
             * Number of solutions in this container.
             */
            uint64_t solutions() const {
                return numSolutions;
            }

            __device__ void setCount(int64_t id, double val) {
                elements[id - minId_] = val;
            }

            GPU_HOST_ATTR void* data() const {
                return elements;
            }
        protected:
            ArraySolution(size_t size, int64_t minId, int64_t maxId, uint64_t numSolutions_, double* elements_) :
                Solution(size, minId, maxId), elements(elements_) {
                numSolutions = numSolutions_;
            };
            double* elements = nullptr;
            uint64_t numSolutions = 0;
    };

    struct TreeNode {
        union {
            uint64_t empty;
            struct {
                uint32_t lowerIdx;
                uint32_t upperIdx;
            } __attribute__((packed));
            double content;
        };

        // delete constructor / destrucor.
        // only create via malloc / free
        TreeNode() = delete;
        ~TreeNode() = delete;
    } __attribute__((packed));

    static_assert(sizeof(TreeNode) == sizeof(uint64_t));

    class TreeSolution: public Solution {
        public:
            TreeSolution(size_t size_, int64_t minId_, int64_t maxId_, int64_t variableCount_) :
                Solution(size_, minId_, maxId_), variableCount(variableCount_)
            {
                tree = (TreeNode*)malloc(size_ * elementSize());
                assert(tree != nullptr);
                memset(tree, 0, size_ * elementSize());
            }

            size_t elementSize() const {
                return sizeof(TreeNode);
            }

            size_t currentTreeSize() const {
                return treeSize;
            }

            size_t variables() const {
                return variableCount;
            }

            void freeData() {
                free(tree);
                tree = nullptr;
            }

            GPU_HOST_ATTR double solutionCountFor(int64_t id) const {
                ulong nextId = 0;
                for (ulong i = 0; i < variableCount; i++) {
                    nextId = ((uint32_t *) &(tree[nextId]))[(id >> (variableCount - i - 1)) & 1];
                    if (nextId == 0) {
                        return 0.0;
                    }
                }
                return tree[nextId].content;
            }

            __device__ void setCount(int64_t id, double value) {
                #if defined(__CUDACC__)
                ulong nextId = 0;
                ulong val = 0;
                if (variableCount == 0) {
                    atomicAdd(&treeSize, 1);
                }
                for (ulong i = 0; i < variableCount; i++) {
                    // lower or upper 32bit, depending on if bit of variable i is set in id
                    uint * lowVal = &((uint *) &(tree[nextId]))[(id >> (variableCount - i - 1)) & 1];
                    // secure our slot by incrementing treeSize
                    if (val == 0 && *lowVal == 0) {
                        val = atomicAdd(&treeSize, 1) + 1;
                    }
                    atomicCAS(lowVal, 0, val);
                    if (*lowVal == val) {
                        if (i < (variableCount - 1)) {
                            val = atomicAdd(&treeSize, 1) + 1;
                        }
                    }
                    nextId = *lowVal;
                }
                tree[nextId].content = value;
                #else
                    fprintf(stderr, "this may not be called from CPU (yet)!");
                    assert(false);
                #endif
            }


            size_t hash() const {
                size_t h = 0;
                hash_combine(h, minId());
                hash_combine(h, maxId());
                hash_combine(h, currentTreeSize());
                if (tree == NULL) {
                    return h;
                }
                hash_combine(h, hashSubtree(tree[0], variableCount));
                return h;
            }

            TreeSolution* gpuCopyWithData(uint64_t* element_buffer) const {
                TreeSolution* gpu = nullptr;
                TreeSolution* cpu = (TreeSolution*)malloc(sizeof(TreeSolution));
                assert(cpu != nullptr);
                memcpy(cpu, this, sizeof(TreeSolution));
                cpu->tree = (TreeNode*)element_buffer;
                gpuErrchk(cudaMalloc(&gpu, elementSize() * dataStructureSize()));
                gpuErrchk(cudaMemcpy(gpu, cpu, sizeof(TreeSolution), cudaMemcpyDeviceToHost));
                free(cpu);
                return gpu;
            };

            /**
             * Copy a solution back from the GPU.
             * Reserve `reserveNodes` additional nodes in the new solution tree.
             */
            static TreeSolution fromGPU(const TreeSolution* gpu, size_t reserveNodes) {
                TreeSolution* cpu = (TreeSolution*)malloc(sizeof(TreeSolution));
                assert(cpu != nullptr);
                gpuErrchk(cudaMemcpy(cpu, gpu, sizeof(TreeSolution), cudaMemcpyDeviceToHost));
                auto size = cpu->treeSize + reserveNodes;
                auto solution = TreeSolution(size, cpu->minId(), cpu->maxId(), cpu->variableCount);
                solution.treeSize = cpu->treeSize;
                memset(solution.tree, 0, solution.elementSize() * size);
                gpuErrchk(cudaMemcpy(solution.tree, cpu->tree, solution.elementSize() * solution.treeSize, cudaMemcpyDeviceToHost));
                return std::move(solution);
            }

            __device__ void incSolutions() {};

            GPU_HOST_ATTR void* data() const {
                return tree;
            }
        protected:
            TreeSolution(size_t size_, int64_t minId_, int64_t maxId_, int64_t variableCount_, int64_t treeSize_, TreeNode* tree_) :
                Solution(size_, minId_, maxId_), variableCount(variableCount_), treeSize(treeSize_), tree(tree_) {};

            size_t hashSubtree(const TreeNode& current, int variables) const {
                if (current.empty == 0) return 0;
                // index cell
                if (variables > 0) {
                    size_t hash = 0;
                    if (current.lowerIdx) {
                        hash_combine(hash, 1);
                        hash_combine(hash, hashSubtree(tree[current.lowerIdx], variables - 1));
                    }
                    if (current.upperIdx) {
                        hash_combine(hash, 2);
                        hash_combine(hash, hashSubtree(tree[current.upperIdx], variables - 1));
                    }
                    return hash;
                } else {
                    return (size_t)current.empty;
                }
            }

            TreeNode* tree = nullptr;
            size_t treeSize = 0;
            size_t variableCount = 0;
    };


    template <class T>
    GPU_HOST_ATTR T dataStructureVisit(
        std::variant<TreeSolution, ArraySolution>& solution,
        T(*treeFunc)(TreeSolution& sol),
        T(*arrayFunc)(ArraySolution& sol)
    ) {
        if (std::holds_alternative<TreeSolution>(solution)) {
            return treeFunc(std::get<TreeSolution>(solution));
        } else if (std::holds_alternative<ArraySolution>(solution)) {
            return arrayFunc(std::get<ArraySolution>(solution));
        }
        __builtin_unreachable();
    }

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
        __builtin_unreachable();
    }

    GPU_HOST_ATTR inline Solution* toSolution(std::variant<TreeSolution, ArraySolution>& solution) {
        return dataStructureVisit<Solution*>(solution,
            [](TreeSolution& sol) { return (Solution*)(&sol); },
            [](ArraySolution& sol) { return (Solution*)(&sol); }
        );
    }

    GPU_HOST_ATTR inline const Solution* toSolution(const std::variant<TreeSolution, ArraySolution>& solution) {
        return dataStructureVisit<const Solution*>(solution,
            [](const TreeSolution& sol) { return (const Solution*)(&sol); },
            [](const ArraySolution& sol) { return (const Solution*)(&sol); }
        );
    }

    /**
     * Create a copy with the data pointer replaced.
     * FIXME: This is a temporary workaround for real
     * resource management.
     */
    inline Solution* gpuCopyWithData(const std::variant<TreeSolution, ArraySolution>& solution, uint64_t* element_buffer) {
        if (const auto sol = std::get_if<ArraySolution>(&solution)) {
                return sol->gpuCopyWithData(element_buffer);
        } else if (const auto sol = std::get_if<TreeSolution>(&solution)) {
                return sol->gpuCopyWithData(element_buffer);
        };
        __builtin_unreachable();
    }

    /***
     * Copy a solution from the GPU and build a variant.
     * @param: reserveElements reserve some elements in case the solution grows.
     *         Currently only used by TreeSolution.
     */
    inline std::variant<TreeSolution, ArraySolution> fromGPU(const Solution* gpu, dataStructure ds, size_t reserveElements) {
        if (ds == ARRAY) {
            // we cannot cast normally here, since `gpu` is only valid on the GPU.
            return std::variant<TreeSolution, ArraySolution>(ArraySolution::fromGPU((ArraySolution*)gpu));
        } else if (ds == TREE) {
            return std::variant<TreeSolution, ArraySolution>(TreeSolution::fromGPU((TreeSolution*)gpu, reserveElements));
        }
        __builtin_unreachable();
    }

    struct GPUVars {
        /// Number of variables.
        size_t count;
        /// Pointer to GPU memory containing the variable buffer.
        int64_t* vars;
    };

    /// type for a bag in the tree decomposition
    struct BagType {
        int64_t correction = 0;
        int64_t exponent = 0;
        int64_t id = 0;
        std::vector<int64_t> variables;
        std::vector<BagType> edges;
        std::vector<std::variant<TreeSolution, ArraySolution>> solution;
        int64_t maxSize = 0;

        size_t hash() const {
            size_t h = 0;
            hash_combine(h, correction);
            hash_combine(h, exponent);
            hash_combine(h, id);
            for (int64_t var : variables) {
                hash_combine(h, var);
            }
            for (const BagType& edge : edges) {
                hash_combine(h, edge.hash());
            }
            for (const auto &sol : solution) {
                hash_combine(h, toSolution(sol)->hash());
            }
            hash_combine(h, maxSize);
            return h;
        }

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

    enum SolveMode {
        DEFAULT = 0,
        ARRAY_TYPE = 1 << 0,
        NO_EXP = 1 << 1,
    };

    typedef struct {
        int64_t minId;
        int64_t maxId;
        SolveMode mode;
    } RunMeta;

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
    inline bool compVars(const int64_t &a, const int64_t &b) {
        return std::abs(a) < std::abs(b);
    }
}

#endif //GPUSAT_TYPES_H_H
