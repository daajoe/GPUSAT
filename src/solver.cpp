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


#include "types.h"
#include "kernel.h"
#include "solver.h"

namespace gpusat {

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



    void Solver::solveProblem(satformulaType &formula, BagType& node, BagType& pnode, nodeType lastNode) {

        std::cerr << "\nsolve problem. isSAT: " << isSat << std::endl;
        std::cerr << " " << node.hash() << std::endl;
        std::cerr << " " << pnode.hash() << std::endl;
        if (isSat > 0) {
            if (node.edges.empty()) {
                BagType cNode;

                double val = 1.0;

                // create initial solution container
                if (solutionType == dataStructure::TREE) {
                    TreeSolution<CpuMem> sol(1, 0, 1, cNode.variables.size());
                    sol.allocate();
                    sol.setCount(0, val);
                    SolutionVariant variant(std::move(sol));
                    cNode.solution.push_back(std::move(variant));
                } else if (solutionType == dataStructure::ARRAY) {
                    ArraySolution<CpuMem> sol(1, 0, 1);
                    sol.allocate();
                    sol.setCount(0, val);
                    SolutionVariant variant(std::move(sol));
                    cNode.solution.push_back(std::move(variant));
                } else {
                    std::cerr << "unknown data structure" << std::endl;
                    exit(1);
                }

                cNode.maxSize = 1;
                solveIntroduceForget(formula, pnode, node, cNode, true, lastNode);
            } else if (node.edges.size() == 1) {
                solveProblem(formula, node.edges[0], node, INTRODUCEFORGET);
                if (isSat == 1) {
                    BagType &cnode = node.edges[0];
                    solveIntroduceForget(formula, pnode, node, cnode, false, lastNode);
                }
            } else if (node.edges.size() > 1) {
                solveProblem(formula, node.edges[0], node, JOIN);
                if (isSat <= 0) {
                    return;
                }

                for (int64_t i = 1; i < node.edges.size(); i++) {
                    BagType& edge1 = node.edges[0];
                    BagType& edge2 = node.edges[i];
                    std::cerr << "\ncombine step SOLVE (" << node.id << ") " << i << " of " << node.edges.size() - 1 << std::endl;
                    std::cerr << edge1.hash() << std::endl;
                    std::cerr << edge2.hash() << std::endl;
                    std::cerr << node.hash() << std::endl;
                    solveProblem(formula, edge2, node, JOIN);
                    if (isSat <= 0) {
                        return;
                    }

                    std::cerr << "\ncombine step JOIN (" << node.id << ") " << i << " of " << node.edges.size() - 1 << std::endl;
                    std::cerr << edge1.hash() << std::endl;
                    std::cerr << edge2.hash() << std::endl;
                    std::cerr << node.hash() << std::endl;


                    std::vector<int64_t> vt;
                    std::set_union(
                            edge1.variables.begin(), edge1.variables.end(),
                            edge2.variables.begin(), edge2.variables.end(),
                            back_inserter(vt));

                    BagType tmp;
                    tmp.variables = vt;
                    std::cerr << tmp.hash() << std::endl;

                    if (i == node.edges.size() - 1) {
                        solveJoin(tmp, edge1, edge2, formula, INTRODUCEFORGET);
                        if (isSat <= 0) {
                            return;
                        }
                        std::cerr << "\ncombine step IF (" << node.id << ") " << i << " of " << node.edges.size() - 1 << std::endl;
                        std::cerr << edge1.hash() << std::endl;
                        std::cerr << edge2.hash() << std::endl;
                        std::cerr << node.hash() << std::endl;

                        node.edges[0] = std::move(tmp);
                        solveIntroduceForget(formula, pnode, node, node.edges[0], false, lastNode);
                    } else {
                        solveJoin(tmp, edge1, edge2, formula, JOIN);
                        node.edges[0] = std::move(tmp);
                    }
                }
            }
        }
    }

    TreeSolution<CpuMem> Solver::arrayToTree(ArraySolution<CpuMem> &table, int64_t size, int64_t numVars, BagType &node, int64_t nextSize) {

        if (table.dataStructureSize() > 0) {

            CudaBuffer<int64_t> buf_exp(&(node.exponent), 1);

            RunMeta meta = {
                .minId = table.minId(),
                .maxId = table.maxId(),
                .mode = solve_mode
            };

            auto tree_gpu = array2treeWrapper(table, buf_exp.data(), numVars, meta);
            table.freeData();

            // make sure we allocate enough space for the next solution container
            size_t reserveNodes = nextSize;
            TreeSolution tree = cpuCopy(tree_gpu, reserveNodes);

            std::cerr << "clean tree num solutions: " << tree.currentTreeSize() << std::endl;
            buf_exp.read(&(node.exponent));

            std::cerr << "tree output hash: " << tree.hash() << std::endl;
            return std::move(tree);
        } else {
            std::cerr << "tree output hash: empty" << std::endl;
            //1 + nextSize : not in use, see above
            return TreeSolution<CpuMem>(0, table.minId(), table.maxId(), numVars);
        }
    }

    TreeSolution<CpuMem> Solver::combineTree(TreeSolution<CpuMem> &t1, TreeSolution<CpuMem> &t2) {
        std::cerr << "combine tree " << t1.hash() << " " << t2.hash() << std::endl;

        // must have the same ID space
        assert(t1.variables() == t2.variables());

        if (t2.dataStructureSize() > 0) {

            // make id spaces equal
            t1.setMinId(std::min(t2.minId(), t1.minId()));
            t2.setMaxId(std::max(t2.maxId(), t1.maxId()));

            // ensure we do not allocate too much space.
            t1.setDataStructureSize(t1.currentTreeSize() + 1);
            t2.setDataStructureSize(t2.currentTreeSize() + 1);

            RunMeta meta = {
                .minId = t2.minId(),
                .maxId = t2.maxId(),
                .mode = solve_mode
            };

            auto result_gpu = combineTreeWrapper(t1, t2, meta);
            t1.freeData();
            t2.freeData();

            TreeSolution<CpuMem> result = cpuCopy(result_gpu);

            std::cerr << "combine tree solutions: " << result.currentTreeSize() << std::endl;
            result.setMinId(std::min(t2.minId(), t1.minId()));
            result.setMaxId(std::max(t2.maxId(), t1.maxId()));
            std::cerr << "combine tree output hash: " << result.hash() << std::endl;
            return std::move(result);
        } else {
            std::cerr << "combine tree output hash: emtpy" << std::endl;
            return TreeSolution<CpuMem>(
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

    void Solver::solveJoin(BagType &node, BagType &edge1, BagType &edge2, satformulaType &formula, nodeType nextNode) {
        /*
        std::cerr << "join input: " << std::endl;
        std::cerr << bagTypeHash(node) << std::endl;
        std::cerr << " " << bagTypeHash(edge1) << std::endl;
        std::cerr << " " << bagTypeHash(edge2) << std::endl;
        */
        isSat = 0;
        this->numJoin++;

        CudaBuffer<int64_t> buf_solVars(node.variables);
        CudaBuffer<int64_t> buf_solVars1(edge1.variables);
        CudaBuffer<int64_t> buf_solVars2(edge2.variables);


        CudaBuffer<double> buf_weights;
        if (formula.variableWeights != nullptr) {
            buf_weights = CudaBuffer<double>(formula.variableWeights, formula.numWeights);
        }

        node.exponent = INT64_MIN;
        CudaBuffer<int64_t> buf_exponent(&(node.exponent), 1);

        uint64_t usedMemory = sizeof(int64_t) * node.variables.size() * 3 + sizeof(int64_t) * edge1.variables.size() + sizeof(int64_t) * edge2.variables.size() + sizeof(double) * formula.numWeights + sizeof(double) * formula.numWeights;

        uint64_t s = sizeof(int64_t);
        uint64_t bagSizeNode = 1;

        if (maxBag > 0) {
            bagSizeNode = 1l << (int64_t) std::min(node.variables.size(), (size_t) maxBag);
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

        for (int64_t _a = 0, run = 0; _a < maxSize; _a++, run++) {
            auto minId = run * bagSizeNode;
            auto maxId = std::min(run * bagSizeNode + bagSizeNode,
                 1ul << (node.variables.size()));
            // FIXME: why + node.variables.size()?
            auto array_size = (maxId - minId) + node.variables.size();
            ArraySolution<CpuMem> tmp_solution(array_size, minId, maxId);


            std::cout << "run: " << run << " " << tmp_solution.minId() << " " << tmp_solution.maxId() << std::endl;

            auto solution_gpu = gpuOwner(tmp_solution);

            for (int64_t b = 0; b < std::max(edge1.solution.size(), edge2.solution.size()); b++) {

                std::optional<SolutionVariant*> edge1_solution = std::nullopt;
                if (b < edge1.solution.size() && hasData(edge1.solution[b])) {
                    edge1_solution = &edge1.solution[b];
                }

                std::optional<SolutionVariant*> edge2_solution = std::nullopt;
                if (b < edge2.solution.size() && hasData(edge2.solution[b])) {
                    edge2_solution = &edge2.solution[b];
                }

                std::cerr << "thread offset: " << minId << " threads " << maxId - minId << std::endl;
                RunMeta meta = {
                    .minId = minId,
                    .maxId = maxId,
                    .mode = solve_mode
                };

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
                    meta
                );
            }

            ArraySolution solution = cpuCopy(solution_gpu);

            std::cerr << "num solutions (join): " << solution.solutions() << std::endl;
            std::cout << "a is " << node.solution.size() << std::endl;

            if (solution.solutions() == 0) {
                solution.freeData();

                if (solutionType == TREE) {
                    if (node.solution.size() > 0) {
                        auto& last = std::get<TreeSolution<CpuMem>>(node.solution.back());
                        last.setMaxId(solution.maxId());
                    } else {
                        auto empty_tree = TreeSolution<CpuMem>(
                            0,
                            solution.minId(),
                            solution.maxId(),
                            node.variables.size()
                        );
                        node.solution.push_back(std::move(empty_tree));
                    }
                } else {
                    node.solution.push_back(std::move(solution));
                }
            } else {
                this->isSat = 1;
                if (solutionType == TREE) {
                    TreeSolution<CpuMem>* last = NULL;
                    if (!node.solution.empty()) {
                        last = &std::get<TreeSolution<CpuMem>>(node.solution.back());
                    }

                    size_t new_max_size = 0;

                    // previous bag is not empty, combine if there is still space.
                    if (last != NULL && last->hasData() && (solution.solutions() + last->currentTreeSize() + 2) < solution.dataStructureSize()) {
                        std::cerr << "first branch" << std::endl;
                        auto tree = arrayToTree(
                            solution,
                            (bagSizeNode) * 2,
                            node.variables.size(),
                            node,
                            last->currentTreeSize() + 1
                        );
                        auto new_tree = combineTree(tree, *last);
                        new_max_size = new_tree.currentTreeSize() + 1;
                        node.solution.back() = std::move(new_tree);
                    // previous back is empty, replace it
                    } else if (last != NULL && !last->hasData()) {
                        std::cerr << "second branch" << std::endl;
                        auto tree = arrayToTree(
                            solution,
                            (bagSizeNode) * 2,
                            node.variables.size(),
                            node,
                            0
                        );
                        tree.setMinId(last->minId());
                        new_max_size = tree.currentTreeSize() + 1;
                        node.solution.back() = std::move(tree);
                    } else {
                        std::cerr << "simple clean tree" << std::endl;
                        auto tree = arrayToTree(
                            solution,
                            (bagSizeNode) * 2,
                            node.variables.size(),
                            node,
                            0
                        );
                        new_max_size = tree.currentTreeSize() + 1;
                        node.solution.push_back(std::move(tree));
                    }
                    node.maxSize = std::max(node.maxSize, (int64_t)new_max_size);
                } else if (solutionType == ARRAY) {
                    // the inital calculated size might overshoot, thus limit
                    // to the solutions IDs we have actually considered.
                    solution.setDataStructureSize((size_t)bagSizeNode);
                    node.maxSize = std::max(node.maxSize, (int64_t)solution.dataStructureSize());
                    node.solution.push_back(std::move(solution));
                }
            }
        }
        if (solutionType == ARRAY) {
            buf_exponent.read(&(node.exponent));
        }
        node.correction = edge1.correction + edge2.correction + edge1.exponent + edge2.exponent;
        int64_t tableSize = 0;
        for (const auto &sol : node.solution) {
            tableSize += dataStructureSize(sol);
        }
        std::cout << "table size: " << tableSize << std::endl;
        this->maxTableSize = std::max(this->maxTableSize, tableSize);
        for (auto &sol : edge1.solution) {
            freeData(sol);
        }
        for (auto &sol : edge2.solution) {
            freeData(sol);
        }
        std::cerr << "JOIN output hash: " << node.hash() << std::endl;
    }

    long qmin(long a, long b, long c, long d) {
        return std::min(a, std::min(b, std::min(c, d)));
    }

    void Solver::solveIntroduceForget(satformulaType &formula, BagType &pnode, BagType &node, BagType &cnode, bool leaf, nodeType nextNode) {

        std::cerr << "IF input hash: " << std::endl;
        std::cerr << "  " << pnode.hash() << std::endl;
        std::cerr << "  " << node.hash() << std::endl;
        std::cerr << "  " << cnode.hash() << std::endl;
        isSat = 0;
        std::vector<int64_t> fVars;
        std::set_intersection(
                node.variables.begin(), node.variables.end(),
                pnode.variables.begin(), pnode.variables.end(),
        std::back_inserter(fVars));
        std::vector<int64_t> iVars = node.variables;
        std::vector<int64_t> eVars = cnode.variables;

        std::cerr << "variables: " << node.variables.size() << std::endl;
        std::cerr << "fvars: " << fVars.size() << std::endl;

        node.variables = fVars;

        this->numIntroduceForget++;

        // get clauses which only contain iVars
        std::vector<int64_t> numVarsClause;
        std::vector<int64_t> clauses;
        int64_t numClauses = 0;
        for (int64_t i = 0; i < formula.clauses.size(); i++) {
            std::vector<int64_t> v;
            std::set_intersection(iVars.begin(), iVars.end(), formula.clauses[i].begin(), formula.clauses[i].end(), back_inserter(v), compVars);
            if (v.size() == formula.clauses[i].size()) {
                numClauses++;
                numVarsClause.push_back(formula.clauses[i].size());
                for (int64_t a = 0; a < formula.clauses[i].size(); a++) {
                    clauses.push_back(formula.clauses[i][a]);
                }
            }
        }

        node.exponent = INT64_MIN;

        CudaBuffer<int64_t> buf_varsE(eVars);
        CudaBuffer<int64_t> buf_varsI(iVars);
        CudaBuffer<int64_t> buf_clauses(clauses);
        CudaBuffer<int64_t> buf_numVarsC(&numVarsClause[0], numClauses);
        CudaBuffer<double> buf_weights(formula.variableWeights, formula.numWeights);
        CudaBuffer<int64_t> buf_exponent(&(node.exponent), 1);

        size_t usedMemory = sizeof(int64_t) * eVars.size() + sizeof(int64_t) * iVars.size() * 3 + sizeof(int64_t) * (clauses.size()) + sizeof(int64_t) * (numClauses) + sizeof(double) * formula.numWeights + sizeof(int64_t) * fVars.size() + sizeof(double) * formula.numWeights;
        uint64_t bagSizeForget = 1;
        uint64_t s = sizeof(int64_t);

        if (maxBag > 0) {
            bagSizeForget = 1ul << (uint64_t) std::min(node.variables.size(), (size_t) maxBag);
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
        std::cerr << "bag size forget: " << bagSizeForget << std::endl;
        std::cerr << "variables: " << node.variables.size() << std::endl;
        std::cerr << "used memory: " << usedMemory << std::endl;

        node.solution.clear();

        for (int64_t _a = 0, run = 0; _a < maxSize; _a++, run++) {

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

            auto solution_gpu = std::visit([](auto& sol) -> std::variant<TreeSolution<CudaMem>, ArraySolution<CudaMem>> {
                return gpuOwner(sol);
            }, solution_tmp);

            for (auto &csol : cnode.solution) {
                if (!hasData(csol)) {
                    continue;
                }

                std::optional<SolutionVariant*> edge_opt = std::nullopt;
                if (!leaf) {
                    edge_opt = &csol;
                }

                // Moved to kernel
                //uint64_t combinations = (uint64_t) pow(2, iVars.size() - fVars.size());
                RunMeta meta = {
                    .minId = sol_minId,
                    .maxId = sol_maxId,
                    .mode = solve_mode
                };

                // FIXME: offset onto global id
                introduceForgetWrapper(
                    solution_gpu,
                    GPUVars {
                        .count = fVars.size(),
                        .vars = buf_varsF.data()
                    },
                    edge_opt,
                    GPUVars {
                        .count = eVars.size(),
                        .vars = buf_varsE.data()
                    },
                    GPUVars {
                        .count = buf_varsI.size(),
                        .vars = buf_varsI.data()
                    },
                    buf_clauses.data(),
                    buf_numVarsC.data(),
                    numClauses,
                    buf_weights.data(),
                    buf_exponent.data(),
                    pow(2, cnode.exponent),
                    meta
                );
            }

            auto solution = std::visit([](auto& gpu_sol) -> SolutionVariant {
                return cpuCopy(gpu_sol);
            }, solution_gpu);

            uint64_t num_entries = 0;
            if (auto sol = std::get_if<TreeSolution<CpuMem>>(&solution)) {
                num_entries = sol->currentTreeSize();
            } else if (auto sol = std::get_if<ArraySolution<CpuMem>>(&solution)) {
                num_entries = sol->solutions();
            } else {
                assert(0);
            }

            std::cerr << "num solutions: " << num_entries << std::endl;
            if (num_entries == 0) {

                freeData(solution);

                if (solutionType == TREE) {
                    if (node.solution.size() > 0) {
                        auto& last = std::get<TreeSolution<CpuMem>>(node.solution.back());
                        auto new_max_id = std::visit(
                            [](auto &s) -> int64_t { return s.maxId(); },
                            solution
                        );
                        last.setMaxId(new_max_id);
                    } else {
                        node.solution.push_back(std::move(solution));
                    }
                } else {
                    node.solution.push_back(std::move(solution));
                }
            } else {
                this->isSat = 1;

                if (solutionType == TREE) {
                    auto sol = std::get<TreeSolution<CpuMem>>(std::move(solution));
                    TreeSolution<CpuMem>* last = NULL;
                    if (!node.solution.empty()) {
                        last = &std::get<TreeSolution<CpuMem>>(node.solution.back());
                    }


                    size_t new_max_size = sol.currentTreeSize() + 1;
                    if (node.variables.size() == 0) {
                        // FIXME:
                        // on nodes with 0 variables, this treeSize indicates satisfiability.
                        // otherwise, it is the index of the last node.
                        // TreeSize is then carried to dataStructureSize.
                        // This mismatch should be solved.
                        // But with 0 variables, we only need one node.
                        new_max_size = 1;
                    }

                    //free(sol.tree);
                    //sol.tree = NULL;

                    if (last != NULL && last->hasData()
                        && (sol.currentTreeSize() + last->currentTreeSize() + 2) < sol.dataStructureSize()) {
                        combineTree(sol, *last);
                        new_max_size = sol.currentTreeSize() + 1;
                        node.solution.back() = std::move(sol);
                    } else {
                        // replace previous bag if needed
                        if (last != NULL && !last->hasData()) {
                            sol.setMinId(last->minId());
                            node.solution.back() = std::move(sol);
                        } else {
                            node.solution.push_back(std::move(sol));
                        }
                    }
                    node.maxSize = std::max(node.maxSize, (int64_t)new_max_size);

                } else if (solutionType == ARRAY) {
                    auto& sol = std::get<ArraySolution<CpuMem>>(solution);
                    // the inital calculated size might overshoot, thus limit
                    // to the solutions IDs we have actually considered.
                    sol.setDataStructureSize((size_t)bagSizeForget);

                    node.solution.push_back(std::move(solution));
                }
            }
        }

        buf_exponent.read(&(node.exponent));
        std::cerr << "exponent: " << node.exponent << std::endl;
        node.correction = cnode.correction + cnode.exponent;
        int64_t tableSize = 0;
        for (const auto &sol : node.solution) {
            tableSize += dataStructureSize(sol);
        }
        std::cout << "table size: " << tableSize << std::endl;
        std::cerr << "IF output hash: " << node.hash() << std::endl;

        this->maxTableSize = std::max(this->maxTableSize, tableSize);
        for (auto &csol : cnode.solution) {
            freeData(csol);
        }
    }
}
