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
#include "solver.h"

namespace gpusat {
    extern void introduceForgetWrapper(
        Solution* solsF,
        GPUVars varsForget,
        const Solution* solsE,
        GPUVars lastVars,
        GPUVars varsIntroduce,
        long *clauses,
        long *numVarsC,
        long numclauses,
        double *weights,
        int64_t *exponent,
        double value,
        RunMeta meta
    );

    extern void solveJoinWrapper(
        ArraySolution *solution,
        const Solution* edge1,
        const Solution* edge2,
        GPUVars variables,
        GPUVars edgeVariables1,
        GPUVars edgeVariables2,
        double *weights,
        double value,
        int64_t *exponent,
        RunMeta meta
    );

    extern void array2treeWrapper(
        TreeSolution *tree,
        const ArraySolution* array,
        int64_t *exponent,
        RunMeta meta
    );

    extern void combineTreeWrapper(
        TreeSolution* to,
        const TreeSolution* from,
        const RunMeta meta
    );

    template <typename T>
    class CudaBuffer {
        private:
            size_t buf_size;

            // prevent c++ from copy assignment.
            // learned this the hard way...
            CudaBuffer(const CudaBuffer& other);
            CudaBuffer& operator=(const CudaBuffer& other);

        public:
            T* device_mem;
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

            ~CudaBuffer();
    };

    template <typename T>
    CudaBuffer<T>::CudaBuffer(T* from, size_t length) {
        T* mem = NULL;
        if (from == NULL) {
            this->buf_size = 0;
        } else {
            gpuErrchk(cudaMalloc((void**)&mem, sizeof(T) * length));
            gpuErrchk(cudaMemcpy(mem, from, sizeof(T) * length, cudaMemcpyHostToDevice));
            this->buf_size = length;
        }
        this->device_mem = mem;
    }

    template <typename T>
    CudaBuffer<T>::CudaBuffer(size_t length) {
        T* mem = NULL;
        gpuErrchk(cudaMalloc((void**)&mem, sizeof(T) * length));
        gpuErrchk(cudaMemset(mem, 0, sizeof(T) * length));
        this->buf_size = length;
        this->device_mem = mem;
    }

    template <typename T>
    CudaBuffer<T>::CudaBuffer(std::vector<T> &vec) {
        T* mem = NULL;
        if (vec.size()) {
            gpuErrchk(cudaMalloc((void**)&mem, sizeof(T) * vec.size()));
            gpuErrchk(cudaMemcpy(mem, &vec[0], sizeof(T) * vec.size(), cudaMemcpyHostToDevice));
            this->buf_size = vec.size();
        } else {
            this->buf_size = 0;
        }
        this->device_mem = mem;
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
        gpuErrchk(cudaMemcpy(to, this->device_mem, sizeof(T) * this->size(), cudaMemcpyDeviceToHost));
    }

    template <typename T>
    CudaBuffer<T>::~CudaBuffer() {
        if (this->device_mem) {
            gpuErrchk(cudaFree(this->device_mem));
        }
        this->device_mem = NULL;
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
                    TreeSolution sol = TreeSolution(1, 0, 1, cNode.variables.size());
                    /*sol.tree = (TreeNode*)malloc(sizeof(TreeNode));
                    sol.tree[0].content = val;
                    sol.maxId = 1,
                    sol.size = 1;
                    */
                    cNode.solution.push_back(std::move(sol));
                } else if (solutionType == dataStructure::ARRAY) {
                    ArraySolution sol = ArraySolution(1, 0, 1);
                    /*sol.elements = (double*)malloc(sizeof(double));
                    sol.elements[0] = val;
                    sol.maxId = 1,
                    sol.size = 1;
                    */
                    cNode.solution.push_back(std::move(sol));
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

    TreeSolution Solver::arrayToTree(ArraySolution &table, int64_t size, int64_t numVars, BagType &node, int64_t nextSize) {
        /*t.numSolutions = 0;
        t.size = size + numVars;
        t.minId = table.minId;
        t.maxId = table.maxId;
        */
        if (table.dataStructureSize() > 0) {

            CudaBuffer<double> buf_sols_old((double*)table.data(), table.dataStructureSize());
            table.freeData();
            //delete [] table.elements;
            //table.elements = NULL;

            size_t max_tree_size = size + numVars;
            CudaBuffer<TreeNode> buf_sols_new(max_tree_size + 2 * numVars);

            //CudaBuffer<uint64_t> buf_num_sol(&(t.solutions()), 1);
            CudaBuffer<int64_t> buf_exp(&(node.exponent), 1);

            double range = table.maxId() - table.minId();

            auto gpu_array = table.gpuCopyWithData((uint64_t*)buf_sols_old.device_mem);

            TreeSolution tmp(max_tree_size, table.minId(), table.maxId(), numVars);
            const auto gpu_tree = tmp.gpuCopyWithData((uint64_t*)buf_sols_new.device_mem);

            int64_t s = std::ceil(range / (1l << 31));
            for (int64_t i = 0; i < s; i++) {
                int64_t id1 = (1 << 31) * i;
                int64_t range = std::min((int64_t) 1 << 31, (int64_t) table.maxId() - table.minId() - (1 << 31) * i);
                RunMeta meta = {
                    .minId = id1,
                    .maxId = id1 + range,
                    .mode = solve_mode
                };
                array2treeWrapper(
                    gpu_tree,
                    gpu_array,
                    buf_exp.device_mem,
                    meta
                );
            }
            // make sure we allocate enough space for the next solution container
            size_t reserveNodes = nextSize + 1;
            TreeSolution t = TreeSolution::fromGPU(gpu_tree, reserveNodes);

            gpuErrchk(cudaFree(gpu_tree));
            gpuErrchk(cudaFree(gpu_array));

            // actually the tree size
            //buf_num_sol.read(&(t.numSolutions));
            std::cerr << "clean tree num solutions (tree size): " << t.currentTreeSize() << ")" << std::endl;
            buf_exp.read(&(node.exponent));

            //int64_t nodeCount = t.treeSize + 1 + nextSize;
            //t.tree = (TreeNode*)malloc(sizeof(TreeNode) * nodeCount);

            //gpuErrchk(cudaMemcpy(t.tree, buf_sols_new.device_mem, sizeof(TreeNode) * nodeCount, cudaMemcpyDeviceToHost));
            t.setMinId(std::min(table.minId(), t.minId()));
            t.setMaxId(std::max(table.maxId(), t.maxId()));
            std::cerr << "tree output hash: " << t.hash() << std::endl;
        } else {
            std::cerr << "tree output hash: empty" << std::endl;
            return TreeSolution(1 + nextSize, table.minId(), table.maxId(), numVars);
        }
    }

    TreeSolution Solver::combineTree(TreeSolution &t1, TreeSolution &t2) {
        std::cerr << "combine tree " << t1.hash() << " " << t2.hash() << std::endl;
        if (t2.dataStructureSize() > 0) {

            CudaBuffer<TreeNode> buf_sols_new((TreeNode*)t1.data(), t1.currentTreeSize() + t2.currentTreeSize() + 2);
            //gpuErrchk(cudaMemset(buf_sols_new.device_mem, 0, buf_sols_new.size() * sizeof(TreeNode)));
            //gpuErrchk(cudaMemcpy(buf_sols_new.device_mem, t.tree, sizeof(TreeNode) * (t.numSolutions + old.numSolutions + 2), cudaMemcpyHostToDevice));

            CudaBuffer<TreeNode> buf_sols_old((TreeNode*)t2.data(), t2.currentTreeSize());

            t2.freeData();
            //free(old.tree);
            //old.tree = NULL;

            //CudaBuffer<uint64_t> buf_num_sol(&t.numSolutions, 1);

            auto gpu_to = t1.gpuCopyWithData((uint64_t*)buf_sols_new.device_mem);
            const auto gpu_from = t2.gpuCopyWithData((uint64_t*)buf_sols_old.device_mem);

            double range = t2.maxId() - t2.minId();
            int64_t s = std::ceil(range / (1l << 31));
            for (long i = 0; i < s; i++) {
                int64_t id1 = (1 << 31) * i;
                int64_t range = std::min((int64_t)1 << 31, t2.maxId() - t2.minId() - (1 << 31) * i);
                RunMeta meta = {
                    .minId = id1,
                    .maxId = id1 + range,
                    .mode = solve_mode
                };
                combineTreeWrapper(
                    gpu_to,
                    gpu_from,
                    meta
                );
            }
            TreeSolution to = TreeSolution::fromGPU(gpu_to, 1);


            gpuErrchk(cudaFree(gpu_to));
            gpuErrchk(cudaFree(gpu_from));

            //buf_num_sol.read(&t.numSolutions);
            std::cerr << "combine tree solutions (tree size): " << to.currentTreeSize() << std::endl;
            //gpuErrchk(cudaMemcpy(t.tree, buf_sols_new.device_mem, sizeof(TreeNode) * (t.numSolutions + 1), cudaMemcpyDeviceToHost));
            to.setMaxId(std::min(t2.minId(), t1.minId()));
            to.setMaxId(std::max(t2.maxId(), t1.maxId()));
            std::cerr << "combine tree output hash: " << to.hash() << std::endl;
            return std::move(to);
        } else {
            std::cerr << "combine tree output hash: emtpy" << std::endl;
            return TreeSolution(
                0,
                std::min(t2.minId(), t1.minId()),
                std::max(t2.maxId(), t1.maxId()),
                t2.variables()
            );
        }
    }

    std::variant<TreeSolution, ArraySolution> initializeSolution(dataStructure ds, int64_t minId, int64_t maxId, size_t size, size_t numVariables) {
        if (ds == TREE) {
            return TreeSolution(size, minId, maxId, numVariables);
        } else if (ds == ARRAY) {
            return ArraySolution(size, minId, maxId);
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


        std::unique_ptr<CudaBuffer<double>> buf_weights( std::make_unique<CudaBuffer<double>>() );
        if (formula.variableWeights != nullptr) {
            buf_weights = std::make_unique<CudaBuffer<double>>(formula.variableWeights, formula.numWeights);
        }

        node.exponent = INT64_MIN;
        CudaBuffer<int64_t> buf_exponent(&(node.exponent), 1);

        int64_t usedMemory = sizeof(int64_t) * node.variables.size() * 3 + sizeof(int64_t) * edge1.variables.size() + sizeof(int64_t) * edge2.variables.size() + sizeof(double) * formula.numWeights + sizeof(double) * formula.numWeights;

        int64_t s = sizeof(int64_t);
        int64_t bagSizeNode = 1;

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

        int64_t maxSize = std::ceil((1l << (node.variables.size())) * 1.0 / bagSizeNode);

        node.solution.clear();

        for (int64_t _a = 0, run = 0; _a < maxSize; _a++, run++) {
            auto minId = run * bagSizeNode;
            auto maxId = std::min(run * bagSizeNode + bagSizeNode,
                 1l << (node.variables.size()));
            // TODO: we do not really need to allocate here.
            ArraySolution tmp_solution((maxId - minId) + node.variables.size(), minId, maxId);


            std::cout << "run: " << run << " " << tmp_solution.minId() << " " << tmp_solution.maxId() << std::endl;

            CudaBuffer<double> buf_sol((double*)tmp_solution.data(), tmp_solution.dataStructureSize());
            auto solution_gpu = tmp_solution.gpuCopyWithData((uint64_t*)buf_sol.device_mem);

            for (int64_t b = 0; b < std::max(edge1.solution.size(), edge2.solution.size()); b++) {

                Solution* edge1_sol = nullptr;
                std::unique_ptr<CudaBuffer<uint64_t>> buf_sol1( std::make_unique<CudaBuffer<uint64_t>>() );
                if (b < edge1.solution.size() && toSolution(edge1.solution[b])->hasData()) {
                    buf_sol1 = std::make_unique<CudaBuffer<uint64_t>>(
                        (uint64_t*)toSolution(edge1.solution[b])->data(),
                        toSolution(edge1.solution[b])->dataStructureSize()
                    );
                    edge1_sol = gpuCopyWithData(edge1.solution[b], buf_sol1->device_mem);
                    //auto copy = gpuSolutionCopy(edge1.solution[b], buf_sol1->device_mem);
                    //edge1_sol = std::move(copy);
                }

                Solution* edge2_sol = nullptr;
                std::unique_ptr<CudaBuffer<uint64_t>> buf_sol2( std::make_unique<CudaBuffer<uint64_t>>() );
                if (b < edge2.solution.size() && toSolution(edge2.solution[b])->hasData()) {
                    buf_sol2 = std::make_unique<CudaBuffer<uint64_t>>(
                        (uint64_t*)toSolution(edge2.solution[b])->data(),
                        toSolution(edge2.solution[b])->dataStructureSize()
                    );
                    edge2_sol = gpuCopyWithData(edge2.solution[b], buf_sol2->device_mem);
                    //auto copy = gpuSolutionCopy(edge2.solution[b], buf_sol2->device_mem);
                    //edge2_sol = std::move(copy);
                }

                int64_t id_offset = minId;
                int64_t threads = maxId - minId;
                std::cerr << "thread offset: " << id_offset << " threads " << threads << std::endl;
                RunMeta meta = {
                    .minId = id_offset,
                    .maxId = id_offset + threads,
                    .mode = solve_mode
                };

                solveJoinWrapper(
                    solution_gpu,
                    edge1_sol,
                    edge2_sol,
                    GPUVars {
                        .count = node.variables.size(),
                        .vars = buf_solVars.device_mem
                    },
                    GPUVars {
                        .count = edge1.variables.size(),
                        .vars = buf_solVars1.device_mem
                    },
                    GPUVars {
                        .count = edge2.variables.size(),
                        .vars = buf_solVars2.device_mem
                    },
                    buf_weights->device_mem,
                    pow(2, edge1.exponent + edge2.exponent),
                    buf_exponent.device_mem,
                    meta
                );

                if (edge1_sol != nullptr) gpuErrchk(cudaFree(edge1_sol));
                if (edge2_sol != nullptr) gpuErrchk(cudaFree(edge2_sol));
            }

            ArraySolution solution = ArraySolution::fromGPU(solution_gpu);
            gpuErrchk(cudaFree(solution_gpu));

            std::cerr << "num solutions (join): " << solution.solutions() << std::endl;
            std::cout << "a is " << node.solution.size() << std::endl;

            if (solution.solutions() == 0) {
                solution.freeData();
                //delete [] solution.elements;
                //solution.elements = NULL;

                if (solutionType == TREE) {
                    if (node.solution.size() > 0) {
                        auto& last = std::get<TreeSolution>(node.solution.back());
                        last.setMinId(solution.maxId());
                    } else {
                        node.solution.push_back(std::move(solution));
                    }
                } else {
                    node.solution.push_back(std::move(solution));
                }
            } else {
                // node.elements is an array here
                //buf_sol.read(solution.elements);
                this->isSat = 1;
                if (solutionType == TREE) {
                    TreeSolution* last = NULL;
                    if (!node.solution.empty()) {
                        last = &std::get<TreeSolution>(node.solution.back());
                    }

                    // previous bag is not empty, combine if there is still space.
                    if (last != NULL && last->hasData() && (solution.solutions() + last->currentTreeSize() + 2) < solution.dataStructureSize()) {
                        std::cerr << "first branch" << std::endl;
                        auto tree = arrayToTree(
                            solution,
                            (bagSizeNode) * 2,
                            node.variables.size(),
                            node,
                            last->dataStructureSize() + 1
                        );
                        auto new_tree = combineTree(tree, *last);
                        //tree.size = tree.numSolutions + 1;
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
                        //tree.size = tree.numSolutions + 1;
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
                        // size?
                        //tree.size = tree.numSolutions + 1;
                        node.solution.push_back(std::move(tree));
                    }
                    node.maxSize = std::max(node.maxSize,
                        (int64_t)toSolution(node.solution.back())->dataStructureSize());
                } else if (solutionType == ARRAY) {
                    //FIXME: check if already fulfilled by construction
                    //solution.size = bagSizeNode;
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
            tableSize += toSolution(sol)->dataStructureSize();
        }
        std::cout << "table size: " << tableSize << std::endl;
        this->maxTableSize = std::max(this->maxTableSize, tableSize);
        for (auto &sol : edge1.solution) {
            toSolution(sol)->freeData();
        }
        for (auto &sol : edge2.solution) {
            toSolution(sol)->freeData();
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
        int64_t bagSizeForget = 1;
        int64_t s = sizeof(int64_t);

        if (maxBag > 0) {
            bagSizeForget = 1l << (int64_t) std::min(node.variables.size(), (size_t) maxBag);
        } else {
            if (solutionType == TREE) {
                if (nextNode == JOIN) {
                    bagSizeForget = qmin(
                        maxMemoryBuffer / s / 2 - 3 * node.variables.size() * sizeof(int64_t),
                        (memorySize - usedMemory - cnode.maxSize * s) / s / 2,
                        (memorySize - usedMemory) / 2 / 3 / s,
                        1l << node.variables.size()
                    );
                } else if (nextNode == INTRODUCEFORGET) {
                    bagSizeForget = qmin(
                        maxMemoryBuffer / s / 2 - 3 * node.variables.size() * sizeof(int64_t),
                        (memorySize - usedMemory - cnode.maxSize * s) / s / 2,
                        (memorySize - usedMemory) / 2 / 2 / s,
                        1l << node.variables.size()
                    );
                }
            } else if (solutionType == ARRAY) {
                bagSizeForget = 1l << (int64_t) std::min(node.variables.size(), (size_t) std::min(log2(maxMemoryBuffer / sizeof(int64_t)), log2(memorySize / sizeof(int64_t) / 3)));
            }
        }

        int64_t maxSize = std::ceil((1l << (node.variables.size())) * 1.0 / bagSizeForget);
        std::cerr << "bag size forget: " << bagSizeForget << std::endl;
        std::cerr << "variables: " << node.variables.size() << std::endl;
        std::cerr << "used memory: " << usedMemory << std::endl;

        node.solution.clear();

        for (int64_t _a = 0, run = 0; _a < maxSize; _a++, run++) {

            int64_t sol_minId =  run * bagSizeForget;
            int64_t sol_maxId = std::min(run * bagSizeForget + bagSizeForget, 1l << (node.variables.size()));

            std::variant<TreeSolution, ArraySolution> solution_tmp = initializeSolution(
                solutionType,
                sol_minId,
                sol_maxId,
                (sol_maxId - sol_minId) * 2 + node.variables.size(),
                node.variables.size()
            );

            CudaBuffer<uint64_t> buf_solsF(
                    (uint64_t*)toSolution(solution_tmp)->data(),
                    toSolution(solution_tmp)->dataStructureSize()
            );
            CudaBuffer<int64_t> buf_varsF(fVars);

            auto solution_gpu = gpuCopyWithData(solution_tmp, buf_solsF.device_mem);

            for (auto &csol : cnode.solution) {
                if (!toSolution(csol)->hasData()) {
                    continue;
                }

                std::unique_ptr<CudaBuffer<uint64_t>> buf_solsE( std::make_unique<CudaBuffer<uint64_t>>() );
                if (!leaf) {
                    buf_solsE = std::make_unique<CudaBuffer<uint64_t>>(
                        (uint64_t*)toSolution(csol)->data(),
                        toSolution(csol)->dataStructureSize()
                    );
                }

                auto edge_gpu = gpuCopyWithData(csol, buf_solsE->device_mem);

                int64_t threads = sol_maxId - sol_minId;
                int64_t id_offset = sol_minId;
                // Moved to kernel
                //uint64_t combinations = (uint64_t) pow(2, iVars.size() - fVars.size());
                RunMeta meta = {
                    .minId = id_offset,
                    .maxId = id_offset + threads,
                    .mode = solve_mode
                };

                // FIXME: offset onto global id
                introduceForgetWrapper(
                    solution_gpu,
                    GPUVars {
                        .count = fVars.size(),
                        .vars = buf_varsF.device_mem
                    },
                    edge_gpu,
                    GPUVars {
                        .count = eVars.size(),
                        .vars = buf_varsE.device_mem
                    },
                    GPUVars {
                        .count = buf_varsI.size(),
                        .vars = buf_varsI.device_mem
                    },
                    buf_clauses.device_mem,
                    buf_numVarsC.device_mem,
                    numClauses,
                    buf_weights.device_mem,
                    buf_exponent.device_mem,
                    pow(2, cnode.exponent),
                    meta
                );
                gpuErrchk(cudaFree(edge_gpu));
            }

            // number of additional tree nodes to reserve
            auto reserveCount = 0;
            if (solutionType == TREE) {
                if (auto last = std::get_if<TreeSolution>(&node.solution.back())) {
                    if (last->hasData()) reserveCount = last->currentTreeSize() + 2;
                };
            }

            auto solution = fromGPU(solution_gpu, solutionType, reserveCount);
            gpuErrchk(cudaFree(solution_gpu));

            auto num_entries = dataStructureVisit<uint64_t>(solution,
                [](const TreeSolution& sol) { return (uint64_t)sol.currentTreeSize(); },
                [](const ArraySolution& sol) { return sol.solutions(); }
            );

            std::cerr << "num solutions: " << num_entries << std::endl;
            if (num_entries == 0) {
                toSolution(solution)->freeData();

                if (solutionType == TREE) {
                    if (node.solution.size() > 0) {
                        auto& last = std::get<TreeSolution>(node.solution.back());
                        last.setMaxId(toSolution(solution)->maxId());
                    } else {
                        node.solution.push_back(std::move(solution));
                    }
                } else {
                    node.solution.push_back(std::move(solution));
                }
            } else {
                this->isSat = 1;

                if (solutionType == TREE) {
                    auto sol = std::get<TreeSolution>(std::move(solution));
                    TreeSolution* last = NULL;
                    if (!node.solution.empty()) {
                        last = &std::get<TreeSolution>(node.solution.back());
                    }


                    if (node.variables.size() == 0) {
                        assert(sol.currentTreeSize() == 0);
                        //sol.numSolutions--;
                    }

                    //free(sol.tree);
                    //sol.tree = NULL;

                    if (last != NULL && last->hasData()
                        && (sol.currentTreeSize() + last->currentTreeSize() + 2) < sol.dataStructureSize()) {

                        //int64_t nodeCount = sol.currentTreeSize() + last->currentTreeSize() + 2;
                        //sol.tree = (TreeNode*)malloc(sizeof(TreeNode) * nodeCount);
                        //cudaMemcpy(sol.tree, buf_solsF.device_mem, sizeof(TreeNode) * nodeCount, cudaMemcpyDeviceToHost);

                        // additional nodes where reserved through `reserveCount`.
                        combineTree(sol, *last);
                        //sol.size = sol.numSolutions + 1;
                        node.solution.back() = std::move(sol);
                    } else {
                        //int64_t nodeCount = sol.numSolutions + 1;
                        //sol.tree = (TreeNode*)malloc(sizeof(TreeNode) * nodeCount);
                        //sol.size = sol.numSolutions + 1;

                        //cudaMemcpy(sol.tree, buf_solsF.device_mem, sizeof(TreeNode) * nodeCount, cudaMemcpyDeviceToHost);

                        // replace previous bag if needed
                        if (last != NULL && !last->hasData()) {
                            sol.setMinId(last->minId());
                            node.solution.back() = std::move(sol);
                        } else {
                            node.solution.push_back(std::move(sol));
                        }
                    }
                    node.maxSize = std::max(node.maxSize,
                        (int64_t)toSolution(node.solution.back())->dataStructureSize());
                } else if (solutionType == ARRAY) {
                    //auto& sol = std::get<ArraySolution>(solution);
                    //sol.size = bagSizeForget;

                    //cudaMemcpy(sol.elements, buf_solsF.device_mem, sizeof(double) * bagSizeForget, cudaMemcpyDeviceToHost);
                    node.solution.push_back(std::move(solution));
                }
            }
        }

        buf_exponent.read(&(node.exponent));
        std::cerr << "exponent: " << node.exponent << std::endl;
        node.correction = cnode.correction + cnode.exponent;
        int64_t tableSize = 0;
        for (const auto &sol : node.solution) {
            tableSize += toSolution(sol)->dataStructureSize();
        }
        std::cout << "table size: " << tableSize << std::endl;
        std::cerr << "IF output hash: " << node.hash() << std::endl;

        this->maxTableSize = std::max(this->maxTableSize, tableSize);
        for (auto &csol : cnode.solution) {
            toSolution(csol)->freeData();
        }
    }
}
