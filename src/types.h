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
#include "alloc.h"

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

    struct TreeNode {
        union {
            uint64_t empty;
            struct {
                uint32_t lowerIdx;
                uint32_t upperIdx;
            } __attribute__((packed));
            double content;
        };

        constexpr TreeNode() : empty(0) {}
    } __attribute__((packed));

    static_assert(sizeof(TreeNode) == sizeof(uint64_t));

    /**
     * CUDA-Compatible RAII manager for solution bags,
     * similar to unique_ptr.
     */
    template <typename T, typename M>
    struct SolutionDataPtr {
        public:
            SolutionDataPtr() {
                data = nullptr;
                allocated_size = 0;
            }
            SolutionDataPtr(T* ptr, size_t size) {
                data = ptr;
                allocated_size = size;
            }
            ~SolutionDataPtr() {
                M()(data, allocated_size);
                data = nullptr;
                allocated_size = 0;
            }

            SolutionDataPtr(const SolutionDataPtr&) = delete;
            SolutionDataPtr& operator=(SolutionDataPtr& other) = delete;

            SolutionDataPtr(SolutionDataPtr&& other) {
                data = other.data;
                allocated_size = other.allocated_size;
                other.data = nullptr;
                other.allocated_size = 0;
            }

            SolutionDataPtr& operator=(SolutionDataPtr&& other) {
                M()(data, allocated_size);
                data = other.data;
                allocated_size = other.allocated_size;
                other.data = nullptr;
                other.allocated_size = 0;
                return *this;
            }

            GPU_HOST_ATTR T* get() const {
                return data;
            }

            GPU_HOST_ATTR T* allocatedSize() const {
                return allocated_size;
            }

        protected:
            T* data;
            size_t allocated_size;
    };

    template <typename M>
    class ArraySolution {
        public:
            ArraySolution(size_t size_, uint64_t minId, uint64_t maxId) :
                size(size_), minId_(minId), maxId_(maxId),
                elements(nullptr, 0), satisfiable(false) {}


            /**
             * Create with data. This will take ownership of gpu_data.
             */
            template<typename M2>
            ArraySolution(const ArraySolution<M2>& other, double* gpu_data, size_t data_size) {
                size = other.dataStructureSize();
                minId_ = other.minId();
                maxId_ = other.maxId();
                satisfiable = other.isSatisfiable();
                // FIXME: this type is technically wrong, but works
                // because no destructors are called on the GPU
                elements = SolutionDataPtr<double, M>(gpu_data, data_size);
            }

            void allocate() {
                elements = SolutionDataPtr<double, M>((double*)M().malloc(size * elementSize()), size * elementSize());
                assert(data() != nullptr);
            }

            void freeData() {
                elements = SolutionDataPtr<double, M>();
            }

            ~ArraySolution() {
                freeData();
            }

            GPU_HOST_ATTR inline uint64_t maxId() const {
                return maxId_;
            }

            GPU_HOST_ATTR inline uint64_t minId() const {
                return minId_;
            }

            GPU_HOST_ATTR inline void setMinId(uint64_t id) {
                minId_ = id;
            }

            GPU_HOST_ATTR inline void setMaxId(uint64_t id) {
                maxId_ = id;
            }

            GPU_HOST_ATTR size_t dataStructureSize() const {
                return size;
            }

            GPU_HOST_ATTR bool isSatisfiable() const {
                return satisfiable;
            }

            GPU_HOST_ATTR void setSatisfiability(bool value) {
                satisfiable = value;
            }

            void setDataStructureSize(size_t new_size) {
                size = new_size;
            }

            ArraySolution(const ArraySolution&) = delete;
            ArraySolution& operator=(ArraySolution& other) = delete;

            // move constructor to ovoid copying solutions
            ArraySolution(ArraySolution&& other) = default;
            // move assignment
            ArraySolution& operator=(ArraySolution&& other) = default;

            constexpr size_t elementSize() const {
                return sizeof(double);
            }

            constexpr double initializer() const {
                return -1.0;
            }

            size_t hash() const {
                size_t h = dataStructureSize();
                hash_combine(h, minId());
                hash_combine(h, maxId());
                hash_combine(h, isSatisfiable());
                if (!hasData()) {
                    return h;
                }
                for (size_t i=0; i < dataStructureSize(); i++) {
                    // regard empty entries as 0.0, for compatibility
                    // with original implementation
                    auto value = std::max(0.0, data()[i]);
                    hash_combine(h, *reinterpret_cast<uint64_t*>(&value));
                }
                return h;
            }

            /**
             * Returns the solution count for a given id.
             * Returns -1.0 if there is no entry for this id.
             * (if initialized correctly)
             */
            GPU_HOST_ATTR double solutionCountFor(int64_t id) const {
                return data()[id - minId_];
            }

            __device__ void setCount(int64_t id, double val) {
                data()[id - minId_] = val;
            }

            GPU_HOST_ATTR double* data() const {
                return elements.get();
            }

            GPU_HOST_ATTR bool hasData() const {
                return data() != nullptr;
            }

        protected:
            size_t size;
            uint64_t minId_;
            uint64_t maxId_;
            SolutionDataPtr<double, M> elements;
            bool satisfiable;
    };

    template <typename M>
    class TreeSolution {
        public:
            TreeSolution(size_t size_, uint64_t minId, uint64_t maxId, size_t variableCount_) :
                size(size_), minId_(minId), maxId_(maxId),
                tree(nullptr, 0), lastNodeIndex(0),
                variableCount(variableCount_), satisfiable(false) {};

            template<typename M2>
            TreeSolution(const TreeSolution<M2>& other, TreeNode* gpu_data, size_t data_size) {
                size = other.dataStructureSize();
                minId_ = other.minId();
                maxId_ = other.maxId();
                lastNodeIndex = other.currentTreeSize() - 1;
                variableCount = other.variables();
                satisfiable = other.isSatisfiable();
                tree = SolutionDataPtr<TreeNode, M>(gpu_data, data_size);
            }

            void allocate() {
                tree = SolutionDataPtr<TreeNode, M>((TreeNode*)M().malloc(size * elementSize()), size * elementSize());
                assert(data() != nullptr);
            }

            constexpr TreeNode initializer() const {
                return TreeNode();
            }

            void freeData() {
                tree = SolutionDataPtr<TreeNode, M>();
            }

            TreeSolution(const TreeSolution&) = delete;
            TreeSolution& operator=(TreeSolution& other) = delete;

            // move constructor to ovoid copying solutions
            TreeSolution(TreeSolution&& other) = default;

            // move assignment
            TreeSolution& operator=(TreeSolution&& other) = default;

            ~TreeSolution() {
                freeData();
            }

            GPU_HOST_ATTR inline uint64_t maxId() const {
                return maxId_;
            }

            GPU_HOST_ATTR inline uint64_t minId() const {
                return minId_;
            }

            GPU_HOST_ATTR inline void setMinId(uint64_t id) {
                minId_ = id;
            }

            GPU_HOST_ATTR inline void setMaxId(uint64_t id) {
                maxId_ = id;
            }

            GPU_HOST_ATTR size_t dataStructureSize() const {
                return size;
            }

            void setDataStructureSize(size_t new_size) {
                size = new_size;
            }

            GPU_HOST_ATTR bool isSatisfiable() const {
                return satisfiable;
            }

            GPU_HOST_ATTR void setSatisfiability(bool value) {
                satisfiable = value;
            }

            constexpr size_t elementSize() const {
                return sizeof(TreeNode);
            }

            size_t currentTreeSize() const {
                return lastNodeIndex + 1;
            }

            GPU_HOST_ATTR size_t variables() const {
                return variableCount;
            }

            /**
             * Returns the solution count for a given id.
             * Returns -1.0 if there is no entry for this id.
             */
            GPU_HOST_ATTR double solutionCountFor(int64_t id) const {
                ulong nextId = 0;
                for (ulong i = 0; i < variables(); i++) {
                    nextId = ((uint32_t *) &(data()[nextId]))[(id >> (variables() - i - 1)) & 1];
                    if (nextId == 0) {
                        return -1.0;
                    }
                }
                return data()[nextId].content;
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
                if (variables() == 0) {
                    setSatisfiability(true);
                }
                for (ulong i = 0; i < variables(); i++) {
                    // lower or upper 32bit, depending on if bit of variable i is set in id
                    uint * lowVal = &((uint *) &(data()[nextId]))[(id >> (variables() - i - 1)) & 1];
                    // secure our slot by incrementing lastNodeIndex
                    if (val == 0 && *lowVal == 0) {
                        val = atomicAdd(&lastNodeIndex, 1) + 1;
                    }
                    atomicCAS(lowVal, 0, val);
                    if (*lowVal == val) {
                        if (i < (variables() - 1)) {
                            val = atomicAdd(&lastNodeIndex, 1) + 1;
                        }
                    }
                    nextId = *lowVal;
                }
                data()[nextId].content = value;
            }


            size_t hash() const {
                size_t h = 0;
                hash_combine(h, minId());
                hash_combine(h, maxId());
                // in the original, nodes with 0 variables cannot be
                // categorized in sat / unsat if their data is freed :/
                hash_combine(h, isSatisfiable() || (variables() == 0));
                if (!hasData()) {
                    return h;
                }
                hash_combine(h, hashSubtree(data()[0], variables()));
                return h;
            }

            GPU_HOST_ATTR TreeNode* data() const {
                return tree.get();
            }

            GPU_HOST_ATTR bool hasData() const {
                return data() != nullptr;
            }
        protected:
            size_t hashSubtree(const TreeNode& current, int variables) const {
                if (current.empty == 0) return 0;
                // index cell
                if (variables > 0) {
                    size_t hash = 0;
                    if (current.lowerIdx) {
                        hash_combine(hash, 1);
                        hash_combine(hash, hashSubtree(data()[current.lowerIdx], variables - 1));
                    }
                    if (current.upperIdx) {
                        hash_combine(hash, 2);
                        hash_combine(hash, hashSubtree(data()[current.upperIdx], variables - 1));
                    }
                    return hash;
                } else {
                    return (size_t)current.empty;
                }
            }

            size_t size;
            uint64_t minId_;
            uint64_t maxId_;
            SolutionDataPtr<TreeNode, M> tree;
            size_t lastNodeIndex;
            size_t variableCount;
            bool satisfiable;
    };

    /**
     * Marks that this object is copied to the GPU and
     * should not manage memory.
     */
    struct GpuOnly {};

    /**
     * Manages memory on the GPU side.
     */
    struct CudaMem {
        void* malloc(size_t size) const {
            uint8_t* mem = nullptr;
            gpuErrchk(cudaMalloc(&mem, size));
            assert(mem != nullptr);
            return mem;
        }
        void operator()(double* ptr, size_t size = 0) const {
            gpuErrchk(cudaFree(ptr));
        }
        void operator()(uint64_t* ptr, size_t size = 0) const {
            gpuErrchk(cudaFree(ptr));
        }
        void operator()(int64_t* ptr, size_t size = 0) const {
            gpuErrchk(cudaFree(ptr));
        }

        void operator()(TreeNode* ptr, size_t size = 0) const {
            gpuErrchk(cudaFree(ptr));
        }
        void operator()(ArraySolution<GpuOnly>* ptr, size_t size = 0) const {
            gpuErrchk(cudaFree(ptr));
        }
        void operator()(TreeSolution<GpuOnly>* ptr, size_t size = 0) const {
            gpuErrchk(cudaFree(ptr));
        }
    };


    extern PinnedSuballocator cuda_pinned_alloc_pool;

    /**
     * Manages memory on the CPU side.
     *
     * Uses Page-Pinned memory, which speeds up
     * Memcpy to the device dramatically.
     */
    struct CpuMem {
        void* malloc(size_t size) const {
            if (size == 0) {
                return nullptr;
            }
            uint8_t* mem = (uint8_t*)cuda_pinned_alloc_pool.allocate(size);
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaDeviceSynchronize());
            return mem;
        }
        void operator()(double* ptr, size_t size) const {
            if (ptr != nullptr) {
                cuda_pinned_alloc_pool.deallocate(ptr, size);
                gpuErrchk(cudaGetLastError());
            }
        }
        void operator()(TreeNode* ptr, size_t size) const {
            if (ptr != nullptr) {
                cuda_pinned_alloc_pool.deallocate(ptr, size);
                gpuErrchk(cudaGetLastError());
            }
        }
    };



    typedef std::variant<TreeSolution<CpuMem>, ArraySolution<CpuMem>> SolutionVariant;
    typedef std::variant<TreeSolution<CudaMem>, ArraySolution<CudaMem>> CudaSolutionVariant;

    inline int64_t dataStructureSize(const SolutionVariant& solution) {
        return std::visit([](const auto& sol) -> int64_t { return sol.dataStructureSize(); }, solution);
    }

    inline void freeData(SolutionVariant& solution) {
        return std::visit([](auto& sol) { return sol.freeData(); }, solution);
    }

    inline bool hasData(SolutionVariant& solution) {
        return std::visit([](auto& sol) { return sol.hasData(); }, solution);
    }

    inline bool isSatisfiable(SolutionVariant& solution) {
        return std::visit([](auto& sol) { return sol.isSatisfiable(); }, solution);
    }

    inline bool isSatisfiable(CudaSolutionVariant& solution) {
        return std::visit([](auto& sol) { return sol.isSatisfiable(); }, solution);
    }

    inline uint64_t minId(CudaSolutionVariant& solution) {
        return std::visit([](auto& sol) { return sol.minId(); }, solution);
    }

    inline uint64_t maxId(CudaSolutionVariant& solution) {
        return std::visit([](auto& sol) { return sol.maxId(); }, solution);
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
        std::vector<SolutionVariant> solution;
        // if all solutions fit in one bag, leave it on the GPU.
        // Solver functions assume this solution is equivalent
        // to the first solution stored in the `solution` vector!
        std::optional<CudaSolutionVariant> cached_solution;
        size_t maxSize = 0;

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
            for (const auto& sol : solution) {
                hash_combine(h, std::visit([](const auto& s) -> size_t {return s.hash(); }, sol));
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
        // number of bags
        int64_t numb = 0;
        // number of variables
        int64_t numVars = 0;
        // tree width
        int64_t width = 0;
        BagType root;
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
        size_t numVars = 0;
        size_t numWeights = 0;
        bool unsat = false;
        double *variableWeights = nullptr;
        std::vector<std::vector<int64_t>> clauses;
        std::vector<int64_t> facts;
    };

    /// the graph type which was the base for the tree decomposition
    enum nodeType {
        JOIN, INTRODUCEFORGET
    };

    typedef struct {
        bool no_exponent = false;
    } SolveConfig;

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
