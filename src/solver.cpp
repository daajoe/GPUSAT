#define GPU_HOST_ATTR

#include <algorithm>
#include <cmath>
#include <iostream>
#include <errno.h>
#include <cuda.h>
#include <memory>
#include <cuda_runtime.h>
#include <signal.h>
#include <optional>


#include "kernel.h"
#include "solver.h"

#define TRACE(fmt, ...) { if (do_trace) {printf("%s() -> ", __func__); printf(fmt, __VA_ARGS__); printf("\n");} }

namespace gpusat {

    size_t BagType::hash() const {
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
        if (cached_solution.has_value()) {
            const auto sol = cpuCopy(cached_solution.value());
            hash_combine(h, std::visit([](const auto& s) -> size_t {return s.hash(); }, sol));
        }
        hash_combine(h, maxSize);
        return h;
    }
    template <typename T>
    class CudaBuffer {
        private:
            size_t buf_size;
            std::unique_ptr<T, CudaMem> device_mem;
        public:
            // prevent c++ from copy assignment.
            CudaBuffer(const CudaBuffer& other) = delete;
            CudaBuffer& operator=(const CudaBuffer& other) = delete;

            CudaBuffer(CudaBuffer&&) = default;
            CudaBuffer& operator=(CudaBuffer&& other) = default;

            T* data() {
                return device_mem.get();
            }

            // creates a buffer with size 0.
            CudaBuffer();
            /// Create an on-device array of T with given length.
            CudaBuffer(size_t length);
            /// Copy a host array of given length to the device.
            CudaBuffer(T* from, size_t length);
            /// Copy a vector to the device.
            /// If the vector is empty, the memory pointer is NULL.
            CudaBuffer(std::vector<T> &vec);
            /// Copy on-device array to `to`
            void read(T* to);
            /// Length of the buffer
            size_t size();

            /// Total memory allocated in bytes.
            size_t mem_size() {
                return size() * sizeof(T);
            }

    };

    template <typename T>
    CudaBuffer<T>::CudaBuffer(T* from, size_t length) {
        T* mem = NULL;
        if (from == NULL) {
            this->buf_size = 0;
        } else {
            mem = (T*)CudaMem().malloc(sizeof(T) * length);
            gpuErrchk(cudaMemcpy(mem, from, sizeof(T) * length, cudaMemcpyHostToDevice));
            this->buf_size = length;
        }
        this->device_mem = std::unique_ptr<T, CudaMem>(mem);
    }

    template <typename T>
    CudaBuffer<T>::CudaBuffer(size_t length) {
        T* mem = (T*)CudaMem().malloc(sizeof(T) * length);
        gpuErrchk(cudaMemset(mem, 0, sizeof(T) * length));
        this->buf_size = length;
        this->device_mem = std::unique_ptr<T, CudaMem>(mem);
    }

    template <typename T>
    CudaBuffer<T>::CudaBuffer(std::vector<T> &vec) {
        T* mem = NULL;
        if (vec.size()) {
            mem = (T*)CudaMem().malloc(sizeof(T) * vec.size());
            gpuErrchk(cudaMemcpy(mem, &vec[0], sizeof(T) * vec.size(), cudaMemcpyHostToDevice));
            this->buf_size = vec.size();
        } else {
            this->buf_size = 0;
        }
        this->device_mem = std::unique_ptr<T, CudaMem>(mem);
    }

    template <typename T>
    CudaBuffer<T>::CudaBuffer() {
        this->device_mem = NULL;
        this->buf_size = 0;
    }

    template <typename T>
    size_t CudaBuffer<T>::size() {
        return this->buf_size;
    }

    template <typename T>
    void CudaBuffer<T>::read(T* to) {
        gpuErrchk(cudaMemcpy(to, this->device_mem.get(), sizeof(T) * this->size(), cudaMemcpyDeviceToHost));
    }


    template<class... Ts> struct sum_visitor : Ts... { using Ts::operator()...; };
    template<class... Ts> sum_visitor(Ts...) -> sum_visitor<Ts...>;

    template<class T>
    static boost::multiprecision::cpp_bin_float_100 solutionSum(T& sol) {
        boost::multiprecision::cpp_bin_float_100 sols = 0.0;
        for (size_t i = sol.minId(); i < sol.maxId(); i++) {
            sols = sols + std::max(sol.solutionCountFor(i), 0.0);
        }
        return sols;
    }

    boost::multiprecision::cpp_bin_float_100 Solver::bagSum(BagType& bag) {
        boost::multiprecision::cpp_bin_float_100 sols = 0.0;
        for (auto& solution : bag.solution) {
            sols = sols + std::visit([](auto& sol) { return solutionSum(sol); }, solution);
        }
        return sols;
    }


    void Solver::solveProblem(const satformulaType &formula, BagType& node, BagType& pnode, nodeType lastNode) {

        TRACE("\n\tis_sat: %d \n\tnode: %lu \n\tpnode: %lu", isSat, node.hash(), pnode.hash());

        if (isSat > 0) {
            if (node.edges.empty()) {
                BagType cNode;

                double val = 1.0;

                // create initial solution container
                if (solutionType == dataStructure::TREE) {
                    TreeSolution<CpuMem> sol(1, 0, 1, cNode.variables.size());
                    sol.allocate();
                    sol.setCount(0, val);
                    // this is not strictly necessary, but ensures
                    // trace conformity with original
                    sol.setSatisfiability(0.0);
                    SolutionVariant variant(std::move(sol));
                    cNode.solution.push_back(std::move(variant));
                } else if (solutionType == dataStructure::ARRAY) {
                    ArraySolution<CpuMem> sol(1, 0, 1);
                    sol.allocate();
                    sol.setCount(0, val);
                    // this is not strictly necessary, but ensures
                    // trace conformity with original
                    sol.setSatisfiability(0.0);
                    SolutionVariant variant(std::move(sol));
                    cNode.solution.push_back(std::move(variant));
                } else {
                    std::cerr << "unknown data structure" << std::endl;
                    exit(1);
                }

                cNode.maxSize = 1;
                solveIntroduceForget(formula, pnode, node, cNode, true, lastNode);
            } else if (node.edges.size() == 1) {
                solveProblem(formula, node.edges.front(), node, INTRODUCEFORGET);
                if (isSat == 1) {
                    BagType &cnode = node.edges.front();
                    solveIntroduceForget(formula, pnode, node, cnode, false, lastNode);
                }
            } else if (node.edges.size() > 1) {
                solveProblem(formula, node.edges.front(), node, JOIN);
                if (isSat <= 0) {
                    return;
                }

                for (auto edge2_it = std::next(node.edges.begin()); edge2_it != node.edges.end(); edge2_it++) {
                    BagType& edge1 = node.edges.front();
                    BagType& edge2 = *edge2_it;

                    TRACE("combine solve(%ld). \n\tedge1: %lu \n\tedge2: %lu, \n\tnode: %lu",
                        node.id, edge1.hash(), edge2.hash(), node.hash());
                    solveProblem(formula, edge2, node, JOIN);
                    if (isSat <= 0) {
                        return;
                    }

                    TRACE("combine join(%ld). \n\tedge1: %lu \n\tedge2: %lu, \n\tnode: %lu",
                        node.id, edge1.hash(), edge2.hash(), node.hash());

                    std::vector<int64_t> vt;
                    std::set_union(
                            edge1.variables.begin(), edge1.variables.end(),
                            edge2.variables.begin(), edge2.variables.end(),
                            back_inserter(vt));

                    BagType tmp;
                    tmp.variables = vt;

                    // last edge
                    if (std::next(edge2_it) == node.edges.end()) {
                        solveJoin(tmp, edge1, edge2, formula, INTRODUCEFORGET);
                        if (isSat <= 0) {
                            return;
                        }

                        TRACE("combine intro/forget(%ld). \n\tedge1: %lu \n\tedge2: %lu, \n\tnode: %lu",
                            node.id, edge1.hash(), edge2.hash(), node.hash());

                        node.edges.front() = std::move(tmp);
                        solveIntroduceForget(formula, pnode, node, node.edges.front(), false, lastNode);
                    } else {
                        solveJoin(tmp, edge1, edge2, formula, JOIN);
                        node.edges.front() = std::move(tmp);
                    }
                }
            }
        }
    }

    TreeSolution<CudaMem> Solver::combineTree(TreeSolution<CudaMem> &t1, TreeSolution<CudaMem> &t2) {

        // must have the same ID space
        assert(t1.variables() == t2.variables());

        if (t2.dataStructureSize() > 0) {

            // make id spaces equal
            t1.setMinId(std::min(t2.minId(), t1.minId()));
            t2.setMaxId(std::max(t2.maxId(), t1.maxId()));

            assert(t1.dataStructureSize() >= t1.currentTreeSize() + t2.currentTreeSize() + 1);

            auto result = combineTreeWrapper(t1, t2);
            t1.freeData();
            t2.freeData();

            TRACE("combine tree size: %lu", result.currentTreeSize());
            result.setMinId(std::min(t2.minId(), t1.minId()));
            result.setMaxId(std::max(t2.maxId(), t1.maxId()));
            return std::move(result);
        } else {
            return TreeSolution<CudaMem>(
                0,
                std::min(t2.minId(), t1.minId()),
                std::max(t2.maxId(), t1.maxId()),
                t2.variables()
            );
        }
    }

    SolutionVariant initializeSolution(dataStructure ds, uint64_t minId, uint64_t maxId, size_t size, size_t numVariables) {
        if (ds == TREE) {
            return TreeSolution<CpuMem>(size, minId, maxId, numVariables);
        } else if (ds == ARRAY) {
            return ArraySolution<CpuMem>(size, minId, maxId);
        }
        __builtin_unreachable();
        std::cout << "unknown data structure: " << ds << std::endl;
        exit(1);
    }

    void Solver::solveJoin(BagType &node, BagType &edge1, BagType &edge2, const satformulaType &formula, nodeType nextNode) {

        isSat = 0;
        this->numJoin++;

        CudaBuffer<int64_t> buf_solVars(node.variables);
        CudaBuffer<int64_t> buf_solVars1(edge1.variables);
        CudaBuffer<int64_t> buf_solVars2(edge2.variables);


        CudaBuffer<double> buf_weights;
        if (formula.variableWeights != nullptr) {
            buf_weights = CudaBuffer<double>(formula.variableWeights, formula.numWeights);
        }

        node.exponent = INT32_MIN;
        CudaBuffer<int32_t> buf_exponent(&(node.exponent), 1);

        uint64_t usedMemory = sizeof(int64_t) * node.variables.size() * 3 + sizeof(int64_t) * edge1.variables.size() + sizeof(int64_t) * edge2.variables.size() + sizeof(double) * formula.numWeights + sizeof(double) * formula.numWeights;

        uint64_t s = sizeof(int64_t);
        uint64_t bagSizeNode = 1;

        if (maxBag > 0) {
            bagSizeNode = 1l << (int64_t) std::min(node.variables.size(), maxBag);
        } else {
            if (solutionType == TREE) {
                if (nextNode == JOIN) {
                    bagSizeNode = std::min((int64_t) (maxMemoryBuffer / s / 2 - node.variables.size() * sizeof(int64_t) * 3), std::min((int64_t) std::min((memorySize - usedMemory - edge1.maxSize * s - edge2.maxSize * s) / s / 2, (memorySize - usedMemory) / 2 / 3 / s), 1l << node.variables.size()));
                } else if (nextNode == INTRODUCEFORGET) {
                    bagSizeNode = std::min((int64_t) (maxMemoryBuffer / s / 2 - node.variables.size() * sizeof(int64_t) * 3), std::min((int64_t) std::min((memorySize - usedMemory - edge1.maxSize * s - edge2.maxSize * s) / s / 2, (memorySize - usedMemory) / 2 / 2 / s), 1l << node.variables.size()));
                }
            } else if (solutionType == ARRAY) {
                bagSizeNode = 1l << (int64_t) std::min(node.variables.size(), (size_t) std::min(log2(maxMemoryBuffer / sizeof(int64_t)), log2(memorySize / sizeof(int64_t) / 3)));
            }
        }

        //bagSizeNode = std::min(1l << 8, bagSizeNode);

        uint64_t maxSize = std::ceil((1l << (node.variables.size())) * 1.0 / bagSizeNode);

        node.solution.clear();

        CudaSolutionVariant solution_gpu(ArraySolution<CudaMem>(0, 0, 0));
        // track wether we can leave the last solution bag on the gpu.
        bool cache_last_solution_bag = true;

        for (size_t _a = 0, run = 0; _a < maxSize; _a++, run++) {
            // save previous soltuion bag
            if (maxId(solution_gpu) > 0) {
                node.solution.push_back(cpuCopy(solution_gpu));
            }

            const auto numVars = node.variables.size();
            const auto minId = run * bagSizeNode;
            const auto maxId = std::min(run * bagSizeNode + bagSizeNode, 1ul << numVars);

            SolutionVariant tmp_solution = [&]() -> SolutionVariant {
                switch (solutionType) {
                    case TREE: {
                        auto tree_size = (maxId - minId) * 2 + numVars;
                        return TreeSolution<CpuMem>(tree_size, minId, maxId, numVars);
                    }
                    case ARRAY: {
                        // FIXME: why + node.variables.size()?
                        auto array_size = (maxId - minId) + numVars;
                        return ArraySolution<CpuMem>(array_size, minId, maxId);
                    }
                }
                assert(0);
                __builtin_unreachable();
            }();

            solution_gpu = gpuOwner(tmp_solution);

            for (size_t b = 0; b < std::max(edge1.solutionBagCount(), edge2.solutionBagCount()); b++) {

                std::optional<CudaSolutionVariant> edge1_solution = std::nullopt;
                if (edge1.cached_solution.has_value()) {
                    // this must be the first and only bag
                    assert(b == 0);
                    assert(edge1.solution.size() == 0);
                    assert(edge1.solutionBagCount() == 1);
                    assert(hasData(edge1.cached_solution.value()));
                    // we need the cached bag again, so we need to write it back.
                    if (maxSize > 1) {
                        edge1.solution.push_back(cpuCopy(edge1.cached_solution.value()));
                    }
                    swap(edge1_solution, edge1.cached_solution);
                } else if (b < edge1.solution.size()) {
                    assert(hasData(edge1.solution[b]));
                    edge1_solution = gpuOwner(edge1.solution[b]);
                }

                std::optional<CudaSolutionVariant> edge2_solution = std::nullopt;
                if (edge2.cached_solution.has_value()) {
                    // this must be the first and only bag
                    assert(b == 0);
                    assert(edge2.solution.size() == 0);
                    assert(edge2.solutionBagCount() == 1);
                    assert(hasData(edge2.cached_solution.value()));
                    // we need the cached bag again, so we need to write it back.
                    if (maxSize > 1) {
                        edge2.solution.push_back(cpuCopy(edge2.cached_solution.value()));
                    }
                    swap(edge2_solution, edge2.cached_solution);
                } else if (b < edge2.solution.size()) {
                    assert(hasData(edge2.solution[b]));
                    edge2_solution = gpuOwner(edge2.solution[b]);
                }

                solveJoinWrapper(
                    solution_gpu,
                    edge1_solution,
                    edge2_solution,
                    GPUVars {
                        .count = node.variables.size(),
                        .vars = buf_solVars.data()
                    },
                    GPUVars {
                        .count = edge1.variables.size(),
                        .vars = buf_solVars1.data()
                    },
                    GPUVars {
                        .count = edge2.variables.size(),
                        .vars = buf_solVars2.data()
                    },
                    buf_weights.data(),
                    pow(2, edge1.exponent + edge2.exponent),
                    buf_exponent.data(),
                    solve_cfg
                );
            }

            TRACE("satisfiable: %d", isSatisfiable(solution_gpu));

            if (!isSatisfiable(solution_gpu)) {

                if (node.solution.size() > 0) {
                    setMaxId(node.solution.back(), maxId);
                    // ensure this solution doesn't get copyied again
                } else {
                    cache_last_solution_bag = false;
                    if (solutionType == TREE) {
                        auto empty_tree = TreeSolution<CpuMem>(0, minId, maxId, numVars);
                        node.solution.push_back(std::move(empty_tree));
                    } else {
                        auto empty_array = ArraySolution<CpuMem>(0, minId, maxId);
                        node.solution.push_back(std::move(empty_array));
                    }
                }
                setMaxId(solution_gpu, 0);
            } else {
                this->isSat = 1;
                if (solutionType == TREE) {
                    TreeSolution<CpuMem>* last = NULL;
                    TreeSolution<CudaMem>& sol = std::get<TreeSolution<CudaMem>>(solution_gpu);
                    if (!node.solution.empty()) {
                        last = &std::get<TreeSolution<CpuMem>>(node.solution.back());
                    }

                    // previous bag is not empty, combine if there is still space.
                    if (last != NULL && last->hasData() && (sol.currentTreeSize() + last->currentTreeSize() + 1) < sol.dataStructureSize()) {
                        TRACE("%s", "first branch");
                        auto gpu_last = gpuOwner(*last);
                        auto new_tree_gpu = combineTree(sol, gpu_last);
                        new_tree_gpu.setDataStructureSize(new_tree_gpu.currentTreeSize());
                        solution_gpu = std::move(new_tree_gpu);
                        node.solution.pop_back();
                    // previous back is empty, replace it
                    } else if (last != NULL && !last->hasData()) {
                        TRACE("%s", "second branch");
                        sol.setMinId(last->minId());
                        sol.setDataStructureSize(sol.currentTreeSize());
                    } else {
                        TRACE("%s", "simple clean tree");
                        sol.setDataStructureSize(sol.currentTreeSize());
                    }
                    cache_last_solution_bag = true;
                    auto max_tree_size = std::get<TreeSolution<CudaMem>>(solution_gpu).currentTreeSize();
                    node.maxSize = std::max(node.maxSize, max_tree_size);
                } else if (solutionType == ARRAY) {
                    auto& sol = std::get<ArraySolution<CudaMem>>(solution_gpu);
                    // the inital calculated size might overshoot, thus limit
                    // to the solutions IDs we have actually considered.
                    sol.setDataStructureSize((size_t)bagSizeNode);
                    node.maxSize = std::max(node.maxSize, sol.dataStructureSize());
                }
            }
        }
        buf_exponent.read(&(node.exponent));

        TRACE("join exponent: %d", node.exponent);

        node.correction = edge1.correction + edge2.correction + edge1.exponent + edge2.exponent;
        size_t tableSize = 0;
        for (const auto &sol : node.solution) {
            tableSize += dataStructureSize(sol);
        }

        this->maxTableSize = std::max(this->maxTableSize, tableSize);
        for (auto &sol : edge1.solution) {
            freeData(sol);
        }
        for (auto &sol : edge2.solution) {
            freeData(sol);
        }

        // if only one solution bag was used, reuse
        // it in the next node
        if (node.solution.size() == 0 && cache_last_solution_bag && do_cache) {
            node.cached_solution = std::make_optional(std::move(solution_gpu));
        } else {
            node.solution.push_back(cpuCopy(solution_gpu));
            node.cached_solution = std::nullopt;
        }

        TRACE("output hash: %lu", node.hash());
    }

    long qmin(long a, long b, long c, long d) {
        return std::min(a, std::min(b, std::min(c, d)));
    }

    void Solver::solveIntroduceForget(const satformulaType &formula, BagType &pnode, BagType &node, BagType &cnode, bool leaf, nodeType nextNode) {

        TRACE("\n\tpnode: %lu \n\tnode: %lu \n\tcnode: %lu\n\tid: %lu", pnode.hash(), node.hash(), cnode.hash(), node.id);
        if (node.variables.size() == 0) {
            std::cout << "node with 0 variables: " << node.id << std::endl;
        }

        isSat = 0;
        std::vector<int64_t> fVars;
        std::set_intersection(
                node.variables.begin(), node.variables.end(),
                pnode.variables.begin(), pnode.variables.end(),
        std::back_inserter(fVars));
        std::vector<int64_t> iVars = node.variables;
        std::vector<int64_t> eVars = cnode.variables;

        node.variables = fVars;

        this->numIntroduceForget++;

        // get clauses which only contain iVars
        std::vector<uint64_t> clauses;
        int64_t numClauses = 0;

        // no facts are considered here, as they should have been eliminated
        // in preprocessing.
        assert(formula.facts.size() == 0);

        for (size_t i = 0; i < formula.clause_offsets.size(); i++) {
            size_t clause_offset = formula.clause_offsets[i];
            size_t c_size = clause_size(formula, i);
            std::vector<int64_t> v;

            auto clause_begin = formula.clause_bag.begin() + clause_offset;
            auto clause_end = formula.clause_bag.begin() + clause_offset + c_size;
            bool applies = std::includes(
                iVars.begin(),
                iVars.end(),
                clause_begin,
                clause_end,
                compVars
            );

            if (applies) {
                numClauses++;
                uint64_t c_vars = 0;
                uint64_t c_signs = 0;
                // assumption: variables in a clause are sorted
                uint64_t variable_index = 0;
                auto lit_iter = formula.clause_bag.begin() + clause_offset;
                auto lit_end = formula.clause_bag.begin() + clause_offset + c_size;
                for (auto var : iVars) {
                    if (abs(*lit_iter) > var) {
                        variable_index++;
                        continue;
                    }
                    assert(abs(*lit_iter) == var);
                    c_vars |= (1ul << variable_index);
                    c_signs |= ((uint64_t)(*lit_iter < 0) << variable_index);
                    lit_iter++;
                    if (lit_iter == lit_end) {
                        break;
                    }
                    variable_index++;
                }
                assert(lit_iter == lit_end);
                clauses.push_back(c_vars);
                clauses.push_back(c_signs);
            }
        }
        //std::cout << node.id << " clauses: " << numClauses << std::endl;

        node.exponent = INT32_MIN;

        CudaBuffer<int64_t> buf_varsE(eVars);
        CudaBuffer<int64_t> buf_varsI(iVars);
        CudaBuffer<uint64_t> buf_clauses(clauses);
        CudaBuffer<double> buf_weights(formula.variableWeights, formula.numWeights);
        CudaBuffer<int32_t> buf_exponent(&(node.exponent), 1);

        size_t usedMemory = sizeof(int64_t) * eVars.size() + sizeof(int64_t) * iVars.size() * 3 + sizeof(int64_t) * (clauses.size()) + sizeof(int64_t) * (numClauses) + sizeof(double) * formula.numWeights + sizeof(int64_t) * fVars.size() + sizeof(double) * formula.numWeights;
        uint64_t bagSizeForget = 1;
        uint64_t s = sizeof(int64_t);

        if (maxBag > 0) {
            bagSizeForget = 1ul << (uint64_t) std::min(node.variables.size(), maxBag);
        } else {
            if (solutionType == TREE) {
                if (nextNode == JOIN) {
                    bagSizeForget = qmin(
                        maxMemoryBuffer / s / 2 - 3 * node.variables.size() * sizeof(int64_t),
                        (memorySize - usedMemory - cnode.maxSize * s) / s / 2,
                        (memorySize - usedMemory) / 2 / 3 / s,
                        1ul << node.variables.size()
                    );
                } else if (nextNode == INTRODUCEFORGET) {
                    bagSizeForget = qmin(
                        maxMemoryBuffer / s / 2 - 3 * node.variables.size() * sizeof(int64_t),
                        (memorySize - usedMemory - cnode.maxSize * s) / s / 2,
                        (memorySize - usedMemory) / 2 / 2 / s,
                        1ul << node.variables.size()
                    );
                }
            } else if (solutionType == ARRAY) {
                bagSizeForget = 1ul << (int64_t) std::min(node.variables.size(), (size_t) std::min(log2(maxMemoryBuffer / sizeof(int64_t)), log2(memorySize / sizeof(int64_t) / 3)));
            }
        }

        uint64_t maxSize = std::ceil((1ul << (node.variables.size())) * 1.0 / bagSizeForget);

        //TRACE("used memory: %lu", usedMemory);

        node.solution.clear();

        CudaSolutionVariant solution_gpu(ArraySolution<CudaMem>(0, 0, 0));
        // track wether we can leave the last solution bag on the gpu.
        bool cache_last_solution_bag = true;

        for (size_t _a = 0, run = 0; _a < maxSize; _a++, run++) {
            // save previous soltuion bag
            if (maxId(solution_gpu) > 0) {
                node.solution.push_back(cpuCopy(solution_gpu));
            }

            uint64_t sol_minId =  run * bagSizeForget;
            uint64_t sol_maxId = std::min(run * bagSizeForget + bagSizeForget, 1ul << (node.variables.size()));

            SolutionVariant solution_tmp = initializeSolution(
                solutionType,
                sol_minId,
                sol_maxId,
                (sol_maxId - sol_minId) * 2 + node.variables.size(),
                node.variables.size()
            );

            CudaBuffer<int64_t> buf_varsF(fVars);

            solution_gpu = gpuOwner(solution_tmp);

            for (int a = 0; a < cnode.solutionBagCount(); a++) {
                std::optional<CudaSolutionVariant> edge_solution = std::nullopt;
                // If we are already given a GPU-resident solution,
                // it must be the first (and only) one.
                if (cnode.cached_solution.has_value()) {
                    assert(cnode.solution.size() == 0);
                    assert(cnode.solutionBagCount() == 1);
                    std::swap(edge_solution, cnode.cached_solution);
                }

                if (!leaf && !edge_solution.has_value()) {
                    auto& csol = cnode.solution[a];
                    assert(hasData(csol));
                    edge_solution = gpuOwner(csol);
                }

                // FIXME: offset onto global id
                introduceForgetWrapper(
                    solution_gpu,
                    GPUVars {
                        .count = fVars.size(),
                        .vars = buf_varsF.data()
                    },
                    edge_solution,
                    GPUVars {
                        .count = eVars.size(),
                        .vars = buf_varsE.data()
                    },
                    GPUVars {
                        .count = buf_varsI.size(),
                        .vars = buf_varsI.data()
                    },
                    buf_clauses.data(),
                    numClauses,
                    buf_weights.data(),
                    buf_exponent.data(),
                    pow(2, cnode.exponent),
                    solve_cfg
                );
            }

            TRACE("satisfiable: %d", isSatisfiable(solution_gpu));

            if (!isSatisfiable(solution_gpu)) {
                if (node.solution.size() > 0) {
                    setMaxId(node.solution.back(), maxId(solution_gpu));
                } else {
                    cache_last_solution_bag = false;
                    if (solutionType == TREE) {
                        auto dummy_tree = TreeSolution<CpuMem>(0, minId(solution_gpu), maxId(solution_gpu), node.variables.size());
                        node.solution.push_back(std::move(dummy_tree));
                    } else {
                        auto dummy_array = ArraySolution<CpuMem>(0, minId(solution_gpu), maxId(solution_gpu));
                        node.solution.push_back(std::move(dummy_array));
                    }
                }
                setMaxId(solution_gpu, 0);
            } else {
                this->isSat = 1;

                if (solutionType == TREE) {
                    auto& sol = std::get<TreeSolution<CudaMem>>(solution_gpu);
                    TreeSolution<CpuMem>* last = NULL;
                    if (!node.solution.empty()) {
                        last = &std::get<TreeSolution<CpuMem>>(node.solution.back());
                    }

                    if (last != NULL && last->hasData()
                        && (sol.currentTreeSize() + last->currentTreeSize() + 1) < sol.dataStructureSize()) {
                        auto gpu_last = gpuOwner(*last);
                        auto new_tree_gpu = combineTree(sol, gpu_last);
                        new_tree_gpu.setDataStructureSize(new_tree_gpu.currentTreeSize());
                        solution_gpu = std::move(new_tree_gpu);
                        node.solution.pop_back();
                    } else {
                        sol.setDataStructureSize(sol.currentTreeSize());

                        // replace previous bag if needed
                        if (last != NULL && !last->hasData()) {
                            sol.setMinId(last->minId());

                            node.solution.pop_back();
                        }
                    }

                    cache_last_solution_bag = true;

                    auto max_tree_size = std::get<TreeSolution<CudaMem>>(solution_gpu).currentTreeSize();
                    node.maxSize = std::max(node.maxSize, max_tree_size);

                } else if (solutionType == ARRAY) {
                    auto& sol = std::get<ArraySolution<CudaMem>>(solution_gpu);
                    // the inital calculated size might overshoot, thus limit
                    // to the solutions IDs we have actually considered.
                    sol.setDataStructureSize(bagSizeForget);
                    cache_last_solution_bag = true;
                }
            }
        }

        buf_exponent.read(&(node.exponent));

        TRACE("node exponent: %d", node.exponent);

        node.correction = cnode.correction + cnode.exponent;
        size_t tableSize = 0;
        for (const auto &sol : node.solution) {
            tableSize += dataStructureSize(sol);
        }

        // if only one solution bag was used, reuse
        // it in the next node
        if (node.solution.size() == 0 && cache_last_solution_bag && do_cache) {
            node.cached_solution = std::make_optional(std::move(solution_gpu));
        } else {
            node.solution.push_back(cpuCopy(solution_gpu));
            node.cached_solution = std::nullopt;
        }

        TRACE("output hash: %lu", node.hash());

        this->maxTableSize = std::max(this->maxTableSize, tableSize);
        for (auto &csol : cnode.solution) {
            freeData(csol);
        }
    }
}
