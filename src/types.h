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

    // internal data of the solution class
    struct SolutionData {
        size_t size = 0;
        int64_t minId = 0;
        int64_t maxId = 0;
    };

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
            GPU_HOST_ATTR virtual int64_t minId() const = 0;

            /**
             * Maximal solution ID of this container.
             */
            GPU_HOST_ATTR virtual int64_t maxId() const = 0;

            /**
             * increase the solution counter by one.
             * FIXME: This is only used in ArraySolution,
             * but is generic because of type erasure in the kernel...
             */
            virtual __device__ void incSolutions() = 0;

            /**
             * Set minimal solution ID of this container.
             */
            GPU_HOST_ATTR virtual void setMinId(int64_t id) = 0;

            /**
             * Set maximal solution ID of this container.
             */
            GPU_HOST_ATTR virtual void setMaxId(int64_t id) = 0;

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
            GPU_HOST_ATTR virtual size_t dataStructureSize() const = 0;

            /**
             * Size of one container element in bytes.
             */
            virtual size_t elementSize() const = 0;


            /**
             * A check sum of the solution container and its properties.
             * FIXME: still based on legacy data structures for comparison.
             */
            virtual size_t hash() const = 0;

            Solution() = default;

            Solution(const Solution&) = delete;
            Solution& operator=(Solution& other) = delete;

            // force operator override in derived classes
            // move constructor to ovoid copying solutions
            Solution(Solution&& other) = delete;
            // move assignment
            Solution& operator=(Solution&& other) = delete;
    };

    struct ArraySolutionData : public SolutionData {
        double* elements = nullptr;
        uint64_t numSolutions = 0;
    };

    class ArraySolution: public Solution {
        public:
            ArraySolution(size_t size, int64_t minId, int64_t maxId) :
                owns_members(true)
            {
                members = new ArraySolutionData();
                members->size = size;
                members->minId = minId;
                members->maxId = maxId;
                members->elements = (double*)malloc(members->size * elementSize());
                assert(members->elements != nullptr);
                for (size_t i=0; i < size; i++) {
                    members->elements[i] = -1.0;
                }
                //memset(members->elements, 0, members->size * elementSize());
            }

            GPU_HOST_ATTR ArraySolution(ArraySolutionData* members_) noexcept : owns_members(false), members(members_) {}

// FIXME: hack in order to allow variants to be trivially
// destructible and thus run on the GPU
#ifndef __CUDACC__
            GPU_HOST_ATTR ~ArraySolution() {
                if (owns_members) {
                    delete members;
                }
            }
#endif

            GPU_HOST_ATTR int64_t maxId() const {
                return members->maxId;
            }

            GPU_HOST_ATTR int64_t minId() const {
                return members->minId;
            }

            GPU_HOST_ATTR void setMinId(int64_t id) {
                members->minId = id;
            }

            GPU_HOST_ATTR void setMaxId(int64_t id) {
                members->maxId = id;
            }

            GPU_HOST_ATTR size_t dataStructureSize() const {
                return members->size;
            }

            void setDataStructureSize(size_t size) {
                members->size = size;
            }

            ArraySolution(const ArraySolution&) = delete;
            ArraySolution& operator=(ArraySolution& other) = delete;

            // move constructor to ovoid copying solutions
            ArraySolution(ArraySolution&& other) : members(other.members), owns_members(other.owns_members) {
                other.owns_members = false;
            };
            // move assignment
            ArraySolution& operator=(ArraySolution&& other) {
                members = other.members;
                owns_members = other.owns_members;
                other.owns_members = false;
                return *this;
            }

            size_t elementSize() const {
                return sizeof(double);
            }

            void freeData() {
                free(members->elements);
                members->elements = nullptr;
            }

            ArraySolutionData* gpuCopyWithData(uint64_t* element_buffer) const {

                ArraySolutionData cpu = *members;
                cpu.elements = (double*)element_buffer;
                ArraySolutionData* gpu = nullptr;
                gpuErrchk(cudaMalloc(&gpu, sizeof(ArraySolutionData)));
                assert(gpu != nullptr);
                gpuErrchk(cudaMemcpy(gpu, &cpu, sizeof(ArraySolutionData), cudaMemcpyHostToDevice));
                return gpu;
            };

            static ArraySolution fromGPU(const ArraySolutionData* gpu) {
                ArraySolutionData tmp;
                gpuErrchk(cudaMemcpy(&tmp, gpu, sizeof(ArraySolutionData), cudaMemcpyDeviceToHost));
                auto solution = ArraySolution(
                    tmp.size,
                    tmp.minId,
                    tmp.maxId
                );
                solution.members->numSolutions = tmp.numSolutions;
                gpuErrchk(cudaMemcpy(solution.members->elements, tmp.elements, solution.elementSize() * solution.dataStructureSize(), cudaMemcpyDeviceToHost));
                return std::move(solution);
            }

            size_t hash() const {
                size_t h = dataStructureSize();
                hash_combine(h, minId());
                hash_combine(h, maxId());
                hash_combine(h, solutions());
                if (members->elements == NULL) {
                    return h;
                }
                for (int i=0; i < dataStructureSize(); i++) {
                    hash_combine(h, *reinterpret_cast<uint64_t*>(&members->elements[i]));
                }
                return h;
            }

            GPU_HOST_ATTR double solutionCountFor(int64_t id) const {
                return members->elements[id - members->minId];
            }

            /**
             * increase the solution counter by one.
             */
            __device__ void incSolutions()  {
#if defined(__CUDACC__)
                atomicAdd(&(members->numSolutions), 1);
#else
                fprintf(stderr, "this may not be called from CPU (yet)!");
                assert(false);
#endif
            }

            __device__ void decSolutions()  {
#if defined(__CUDACC__)
                atomicSub(&(members->numSolutions), 1);
#else
                fprintf(stderr, "this may not be called from CPU (yet)!");
                assert(false);
#endif
            }

            /**
             * Number of solutions in this container.
             */
            uint64_t solutions() const {
                return members->numSolutions;
            }

            __device__ void setCount(int64_t id, double val) {
                members->elements[id - members->minId] = val;
            }

            GPU_HOST_ATTR void* data() const {
                return members->elements;
            }
        protected:
            ArraySolutionData* members;
            // FIXME: if this is const, the default move assignment cannot
            // be derived
            bool owns_members;
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

    struct TreeSolutionData : public SolutionData {
        TreeNode* tree = nullptr;
        size_t treeSize = 0;
        size_t variableCount = 0;
    };

    class TreeSolution: public Solution {
        public:
            TreeSolution(size_t size_, int64_t minId_, int64_t maxId_, int64_t variableCount_) :
                owns_members(true)
            {
                members = new TreeSolutionData();
                members->size = size_;
                members->minId = minId_;
                members->maxId = maxId_;
                members->variableCount = variableCount_;
                members->tree = (TreeNode*)malloc(members->size * elementSize());
                assert(members->tree != nullptr);
                memset(members->tree, 0, members->size * elementSize());
                members->treeSize = 0;
            }

            TreeSolution(const TreeSolution&) = delete;
            TreeSolution& operator=(TreeSolution& other) = delete;

            // move constructor to ovoid copying solutions
            TreeSolution(TreeSolution&& other) : members(other.members), owns_members(other.owns_members) {
                other.owns_members = false;
            };
            // move assignment
            TreeSolution& operator=(TreeSolution&& other) {
                members = other.members;
                owns_members = other.owns_members;
                other.owns_members = false;
                return *this;
            }

            GPU_HOST_ATTR TreeSolution(TreeSolutionData* members_) noexcept : owns_members(false), members(members_) {}

// FIXME: hack in order to allow variants to be trivially
// destructible and thus run on the GPU
#ifndef __CUDACC__
            GPU_HOST_ATTR ~TreeSolution() {
                if (owns_members) {
                    delete members;
                }
            }
#endif

            GPU_HOST_ATTR int64_t maxId() const {
                return members->maxId;
            }

            GPU_HOST_ATTR int64_t minId() const {
                return members->minId;
            }

            GPU_HOST_ATTR void setMinId(int64_t id) {
                members->minId = id;
            }

            GPU_HOST_ATTR void setMaxId(int64_t id) {
                members->maxId = id;
            }

            GPU_HOST_ATTR size_t dataStructureSize() const {
                return members->size;
            }

            size_t elementSize() const {
                return sizeof(TreeNode);
            }

            /**
             * FIXME: This is the *INDEX* of the largest element
             * -> make this more clear and get rid of magic +1s
             */
            size_t currentTreeSize() const {
                return members->treeSize;
            }

            size_t variables() const {
                return members->variableCount;
            }

            void freeData() {
                free(members->tree);
                members->tree = nullptr;
            }

            GPU_HOST_ATTR double solutionCountFor(int64_t id) const {
                ulong nextId = 0;
                for (ulong i = 0; i < members->variableCount; i++) {
                    nextId = ((uint32_t *) &(members->tree[nextId]))[(id >> (members->variableCount - i - 1)) & 1];
                    if (nextId == 0) {
                        return 0.0;
                    }
                }
                return members->tree[nextId].content;
            }

            /**
             * FIXME: for now, this is not thread-safe on the CPU!
             */
            __device__ void setCount(int64_t id, double value) {
#ifndef __CUDACC__
                auto atomicAdd = [](uint64_t* address, uint64_t val) -> uint64_t {
                    auto old = *address;
                    *address += val;
                    return old;
                };
                auto atomicCAS = [](uint* address, ulong compare, ulong val) -> ulong {
                    auto old = *address;
                    *address = (old == compare ? val : old);
                    return old;
                };
#endif

                ulong nextId = 0;
                ulong val = 0;
                if (members->variableCount == 0) {
                    atomicAdd(&members->treeSize, 1);
                }
                for (ulong i = 0; i < members->variableCount; i++) {
                    // lower or upper 32bit, depending on if bit of variable i is set in id
                    uint * lowVal = &((uint *) &(members->tree[nextId]))[(id >> (members->variableCount - i - 1)) & 1];
                    // secure our slot by incrementing treeSize
                    if (val == 0 && *lowVal == 0) {
                        val = atomicAdd(&members->treeSize, 1) + 1;
                    }
                    atomicCAS(lowVal, 0, val);
                    if (*lowVal == val) {
                        if (i < (members->variableCount - 1)) {
                            val = atomicAdd(&members->treeSize, 1) + 1;
                        }
                    }
                    nextId = *lowVal;
                }
                members->tree[nextId].content = value;
            }


            size_t hash() const {
                size_t h = 0;
                hash_combine(h, minId());
                hash_combine(h, maxId());
                // FIXME:
                // on nodes with 0 variables, this treeSize indicates satisfiability.
                // otherwise, it is the index of the last node.
                // This mismatch should be solved.
                hash_combine(h, currentTreeSize() - (members->variableCount == 0));
                if (members->tree == NULL) {
                    return h;
                }
                hash_combine(h, hashSubtree(members->tree[0], members->variableCount));
                return h;
            }

            TreeSolutionData* gpuCopyWithData(uint64_t* element_buffer) const {
                TreeSolutionData* gpu = nullptr;
                TreeSolutionData cpu = *members;
                cpu.tree = (TreeNode*)element_buffer;
                gpuErrchk(cudaMalloc(&gpu, sizeof(TreeSolutionData)));
                assert(gpu != nullptr);
                gpuErrchk(cudaMemcpy(gpu, &cpu, sizeof(TreeSolutionData), cudaMemcpyHostToDevice));
                return gpu;
            };

            /**
             * Copy a solution back from the GPU.
             * Reserve `reserveNodes` additional nodes in the new solution tree.
             */
            static TreeSolution fromGPU(const TreeSolutionData* gpu, size_t reserveNodes) {
                TreeSolutionData tmp;
                gpuErrchk(cudaMemcpy(&tmp, gpu, sizeof(TreeSolutionData), cudaMemcpyDeviceToHost));
                // +1, because treeSize is INDEX of the last node
                auto size = tmp.treeSize + reserveNodes + 1;
                auto solution = TreeSolution(
                        size,
                        tmp.minId,
                        tmp.maxId,
                        tmp.variableCount
                );
                solution.members->treeSize = tmp.treeSize;
                // FIXME: only memset elements that are not copied over
                //memset(solution.members->tree, 0, solution.elementSize() * size);
                gpuErrchk(cudaMemcpy(solution.members->tree, tmp.tree, solution.elementSize() * size, cudaMemcpyDeviceToHost));
                return std::move(solution);
            }

            __device__ void incSolutions() {};

            GPU_HOST_ATTR void* data() const {
                return members->tree;
            }
        protected:
            size_t hashSubtree(const TreeNode& current, int variables) const {
                if (current.empty == 0) return 0;
                // index cell
                if (variables > 0) {
                    size_t hash = 0;
                    if (current.lowerIdx) {
                        hash_combine(hash, 1);
                        hash_combine(hash, hashSubtree(members->tree[current.lowerIdx], variables - 1));
                    }
                    if (current.upperIdx) {
                        hash_combine(hash, 2);
                        hash_combine(hash, hashSubtree(members->tree[current.upperIdx], variables - 1));
                    }
                    return hash;
                } else {
                    return (size_t)current.empty;
                }
            }

            TreeSolutionData* members;
            // FIXME: if this is const, the default move assignment cannot
            // be derived
            bool owns_members;
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
        assert(0);
#ifndef __CUDACC__
        __builtin_unreachable();
#endif
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
        assert(0);
#ifndef __CUDACC__
        __builtin_unreachable();
#endif
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
    inline std::variant<TreeSolutionData*,ArraySolutionData*> gpuCopyWithData(const std::variant<TreeSolution, ArraySolution>& solution, uint64_t* element_buffer) {
        if (const auto sol = std::get_if<ArraySolution>(&solution)) {
            return std::variant<TreeSolutionData*,ArraySolutionData*>(sol->gpuCopyWithData(element_buffer));
        } else if (const auto sol = std::get_if<TreeSolution>(&solution)) {
            return std::variant<TreeSolutionData*,ArraySolutionData*>(sol->gpuCopyWithData(element_buffer));
        };
        __builtin_unreachable();
    }

    inline void freeGPUDataVariant(std::variant<TreeSolutionData*,ArraySolutionData*>& data) {
        if (std::holds_alternative<TreeSolutionData*>(data)) {
            gpuErrchk(cudaFree(std::get<TreeSolutionData*>(data)));
            return;
        } else if (std::holds_alternative<ArraySolutionData*>(data)) {
            gpuErrchk(cudaFree(std::get<ArraySolutionData*>(data)));
            return;
        }
        assert(false);
    }

    /***
     * Copy a solution from the GPU and build a variant.
     * @param: reserveElements reserve some elements in case the solution grows.
     *         Currently only used by TreeSolution.
     */
    inline std::variant<TreeSolution, ArraySolution> fromGPU(const std::variant<TreeSolutionData*,ArraySolutionData*>& gpu_data, size_t reserveElements) {
        if (std::holds_alternative<ArraySolutionData*>(gpu_data)) {
            auto ptr = std::get<ArraySolutionData*>(gpu_data);
            // we cannot cast normally here, since `gpu` is only valid on the GPU.
            ArraySolution solution = ArraySolution::fromGPU(ptr);
            return std::variant<TreeSolution, ArraySolution>(std::move(solution));
        } else if (std::holds_alternative<TreeSolutionData*>(gpu_data)) {
            auto ptr = std::get<TreeSolutionData*>(gpu_data);
            TreeSolution solution = TreeSolution::fromGPU(ptr, reserveElements);
            return std::variant<TreeSolution, ArraySolution>(std::move(solution));
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
