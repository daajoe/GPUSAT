#define GPU_HOST_ATTR

#include <algorithm>
#include <cmath>
#include <iostream>
#include <solver.h>
#include <errno.h>
#include <cuda.h>
#include <memory>
#include <types.h>
#include <cuda_runtime.h>
#include <signal.h>

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

using gpusat::SolveMode;
using gpusat::GPUVars;

extern void introduceForgetWrapper(long *solsF,  GPUVars varsF,  long *solsE,  GPUVars lastVars, long combinations, long minIdE, long maxIdE, long startIDF, long startIDE,  long *sols, GPUVars varsIntroduce, long *clauses,  long *numVarsC, long numclauses,  double *weights,  long *exponent, double value, size_t threads, long id_offset, SolveMode mode);

    extern void solveJoinWrapper( long *solutions,  long *edge1,  long *edge2,  GPUVars variables, GPUVars edgeVariables1,  GPUVars edgeVariables2, long minId1, long maxId1, long minId2, long maxId2, long startIDNode, long startIDEdge1, long startIDEdge2,  double *weights,  long *sols, double value,  long *exponent, size_t threads, long id_offset, SolveMode mode);

    extern void array2treeWrapper(long numVars, long *tree, double *solutions_old, long *treeSize, long startId, long *exponent, size_t threads, long id_offset, SolveMode mode);

    extern void combineTreeWrapper(long numVars, long *tree, long *solutions_old, long *treeSize, long startId, size_t threads, long id_offset);


namespace gpusat {

    size_t hashSubtree(const TreeNode* tree, TreeNode current, int variables) {
        if (current.empty == 0) return 0;
        // index cell
        if (variables > 0) {
            size_t hash = 0;
            if (current.lowerIdx) {
                hash ^= 1;
                hash ^= (hashSubtree(tree, tree[current.lowerIdx], variables - 1) << 1);
            }
            if (current.upperIdx) {
                hash ^= 2;
                hash ^= (hashSubtree(tree, tree[current.upperIdx], variables - 1) << 1);
            }
            return hash;
        } else {
            return (size_t)current.empty;
        }
    }
    size_t hashArray(const ArraySolution& solution) {
        size_t h = 0;
        for (int i=0; i < solution.size; i++) {
            h ^= (*reinterpret_cast<uint64_t*>(&solution.elements[i]) << 1);
        }
        return h;
    }

    size_t treeTypeHash(const TreeSolution& t, int vars) {
        size_t h = 0;
        h = h ^ (t.minId << 1);
        h = h ^ (t.maxId << 1);
        h = h ^ (t.numSolutions << 1);
        if (t.tree == NULL) {
            return h;
        }
        return h = h ^ (hashSubtree(t.tree, t.tree[0], vars) << 1);
    }
    size_t hashSolution(const std::variant<TreeSolution, ArraySolution>& sol, int vars) {
        if (auto t = std::get_if<TreeSolution>(&sol)) {
            return treeTypeHash(*t, vars);
        } else if (auto a = std::get_if<ArraySolution>(&sol)) {
            return hashArray(*a);
        }
        std::cerr << "unknown solution data structure" << std::endl;
        exit(1);
    }

    size_t bagTypeHash(const BagType& input) {
        size_t h = 0;
        h = h ^ input.correction << 1;
        h = h ^ input.exponent << 1;
        h = h ^ input.id << 1;
        for (cl_long var : input.variables) {
            h = (h ^ var) << 1;
        }
        for (const BagType& edge : input.edges) {
            h = h ^ (bagTypeHash(edge) << 1);
        }
        for (const auto &sol : input.solution) {
            h = h ^ (hashSolution(sol, input.variables.size()) << 1);
        }
        h = h ^ (input.maxSize << 1);
        return h;
    }

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
        std::cerr << " " << bagTypeHash(node) << std::endl;
        std::cerr << " " << bagTypeHash(pnode) << std::endl;
        if (isSat > 0) {
            if (node.edges.empty()) {
                BagType cNode;

                double val = 1.0;

                // create initial solution container
                if (solutionType == dataStructure::TREE) {
                    TreeSolution sol = TreeSolution();
                    sol.tree = new TreeNode[1];
                    sol.tree[0].content = val;
                    sol.maxId = 1,
                    sol.size = 1;
                    cNode.solution.push_back(std::move(sol));
                } else if (solutionType == dataStructure::ARRAY) {
                    ArraySolution sol = ArraySolution();
                    sol.elements = new double[1];
                    sol.elements[0] = val;
                    sol.maxId = 1,
                    sol.size = 1;
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
                BagType tmp;
                BagType& edge1 = node.edges[0];

                bool first = false;
                for (cl_long i = 1; i < node.edges.size(); i++) {
                    BagType& edge2 = node.edges[i];
                    std::cerr << "\ncombine step SOLVE (" << node.id << ") " << i << " of " << node.edges.size() - 1 << std::endl;
                    std::cerr << bagTypeHash(edge1) << std::endl;
                    std::cerr << bagTypeHash(edge2) << std::endl;
                    std::cerr << bagTypeHash(node) << std::endl;
                    solveProblem(formula, edge2, node, JOIN);
                    if (isSat <= 0) {
                        return;
                    }

                    std::cerr << "\ncombine step JOIN (" << node.id << ") " << i << " of " << node.edges.size() - 1 << std::endl;
                    std::cerr << bagTypeHash(edge1) << std::endl;
                    std::cerr << bagTypeHash(edge2) << std::endl;
                    std::cerr << bagTypeHash(node) << std::endl;


                    std::vector<cl_long> vt;
                    std::set_union(
                            edge1.variables.begin(), edge1.variables.end(),
                            edge2.variables.begin(), edge2.variables.end(),
                            back_inserter(vt));
                    tmp.variables = vt;

                    if (i == node.edges.size() - 1) {
                        solveJoin(tmp, edge1, edge2, formula, INTRODUCEFORGET);
                        if (isSat <= 0) {
                            return;
                        }
                        std::cerr << "\ncombine step IF (" << node.id << ") " << i << " of " << node.edges.size() - 1 << std::endl;
                        std::cerr << bagTypeHash(edge1) << std::endl;
                        std::cerr << bagTypeHash(edge2) << std::endl;
                        std::cerr << bagTypeHash(node) << std::endl;

                        node.edges[0] = std::move(tmp);
                        solveIntroduceForget(formula, pnode, node, node.edges[0], false, lastNode);
                    } else {
                        solveJoin(tmp, edge1, edge2, formula, JOIN);
                        if (!first) {
                            std::move(node.edges[0]);
                            node.edges[0] = std::move(tmp);
                            first = true;
                        }
                    }
                }
            }
        }
    }

    TreeSolution Solver::arrayToTree(ArraySolution &table, int64_t size, int64_t numVars, BagType &node, int64_t nextSize) {
        TreeSolution t;
        t.numSolutions = 0;
        t.size = size + numVars;
        t.minId = table.minId;
        t.maxId = table.maxId;
        if (table.size > 0) {

            CudaBuffer<double> buf_sols_old(table.elements, table.size);
            delete [] table.elements;
            table.elements = NULL;

            CudaBuffer<int64_t> buf_sols_new(t.size + 2 * numVars);

            CudaBuffer<int64_t> buf_num_sol(&(t.numSolutions), 1);
            CudaBuffer<int64_t> buf_exp(&(node.exponent), 1);

            cl_long error1 = 0, error2 = 0;
            cl_double range = table.maxId - table.minId;
            cl_long s = std::ceil(range / (1l << 31));
            for (cl_long i = 0; i < s; i++) {
                cl_long id1 = (1 << 31) * i;
                cl_long range = std::min((cl_long) 1 << 31, (cl_long) table.maxId - table.minId - (1 << 31) * i);
                array2treeWrapper(
                    numVars,
                    (int64_t*)buf_sols_new.device_mem,
                    buf_sols_old.device_mem,
                    buf_num_sol.device_mem,
                    table.minId,
                    buf_exp.device_mem,
                    range,
                    id1,
                    solve_mode
                );
            }
            // actually the tree size
            buf_num_sol.read(&(t.numSolutions));
            std::cerr << "clean tree num solutions: " << t.numSolutions << std::endl;
            buf_exp.read(&(node.exponent));

            int64_t nodeCount = t.numSolutions + 1 + nextSize;
            t.tree = new TreeNode[nodeCount];
            gpuErrchk(cudaMemcpy(t.tree, buf_sols_new.device_mem, sizeof(TreeNode) * nodeCount, cudaMemcpyDeviceToHost));
        }
        t.size = (t.numSolutions + 1 + nextSize);
        t.minId = std::min(table.minId, t.minId);
        t.maxId = std::max(table.maxId, t.maxId);
        std::cerr << "tree output hash: " << treeTypeHash(t, numVars) << std::endl;
        return t;
    }

    void Solver::combineTree(TreeSolution &t, TreeSolution &old, cl_long numVars) {
        std::cerr << "combine tree " << treeTypeHash(t, numVars) << " " << treeTypeHash(old, numVars) << std::endl;
        if (old.size > 0) {

            CudaBuffer<TreeNode> buf_sols_new(t.tree, t.numSolutions + old.numSolutions + 2);
            gpuErrchk(cudaMemset(buf_sols_new.device_mem, 0, buf_sols_new.size() * sizeof(TreeNode)));
            gpuErrchk(cudaMemcpy(buf_sols_new.device_mem, t.tree, sizeof(TreeNode) * (t.numSolutions + old.numSolutions + 2), cudaMemcpyHostToDevice));

            CudaBuffer<TreeNode> buf_sols_old(old.tree, old.size);
            delete [] old.tree;
            old.tree = NULL;

            CudaBuffer<cl_long> buf_num_sol(&t.numSolutions, 1);

            cl_double range = old.maxId - old.minId;
            cl_long s = std::ceil(range / (1l << 31));
            for (long i = 0; i < s; i++) {
                cl_long id1 = (1 << 31) * i;
                cl_long range = std::min((int64_t)1 << 31, old.maxId - old.minId - (1 << 31) * i);
                combineTreeWrapper(
                    numVars,
                    (int64_t*)buf_sols_new.device_mem,
                    (int64_t*)buf_sols_old.device_mem,
                    buf_num_sol.device_mem,
                    old.minId,
                    range,
                    id1
                );
            }
            buf_num_sol.read(&t.numSolutions);
            std::cerr << "combine tree solutions: " << t.numSolutions << std::endl;
            gpuErrchk(cudaMemcpy(t.tree, buf_sols_new.device_mem, sizeof(TreeNode) * (t.numSolutions + 1), cudaMemcpyDeviceToHost));
        }
        t.minId = std::min(old.minId, t.minId);
        t.maxId = std::max(old.maxId, t.maxId);
        std::cerr << "combine tree output hash: " << treeTypeHash(t, numVars) << std::endl;
    }

    std::variant<TreeSolution, ArraySolution> initializeSolution(dataStructure ds, int64_t minId, int64_t maxId, size_t size) {
        if (ds == TREE) {
            TreeSolution solution;
            solution.minId = minId;
            solution.numSolutions = 0;
            solution.maxId = maxId;
            solution.size = size;
            solution.tree = new TreeNode[solution.size];
            for (size_t i=0; i < solution.size; i++) {
                solution.tree[i].empty = 0;
            }
            return std::move(solution);
        } else if (ds == ARRAY) {
            ArraySolution solution;
            solution.minId = minId;
            solution.numSolutions = 0;
            solution.maxId = maxId;
            solution.size = size;
            solution.elements = new double[solution.size];
            for (size_t i=0; i < solution.size; i++) {
                ((int64_t*)solution.elements)[i] = 0;
            }
            return std::move(solution);
        }
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

        CudaBuffer<cl_long> buf_solVars(node.variables);
        CudaBuffer<cl_long> buf_solVars1(edge1.variables);
        CudaBuffer<cl_long> buf_solVars2(edge2.variables);


        std::unique_ptr<CudaBuffer<cl_double>> buf_weights( std::make_unique<CudaBuffer<cl_double>>() );
        if (formula.variableWeights != nullptr) {
            buf_weights = std::make_unique<CudaBuffer<cl_double>>(formula.variableWeights, formula.numWeights);
        }

        node.exponent = CL_LONG_MIN;
        CudaBuffer<cl_long> buf_exponent(&(node.exponent), 1);

        cl_long usedMemory = sizeof(cl_long) * node.variables.size() * 3 + sizeof(cl_long) * edge1.variables.size() + sizeof(cl_long) * edge2.variables.size() + sizeof(cl_double) * formula.numWeights + sizeof(cl_double) * formula.numWeights;

        cl_long s = sizeof(cl_long);
        cl_long bagSizeNode = 1;

        if (maxBag > 0) {
            bagSizeNode = 1l << (cl_long) std::min(node.variables.size(), (size_t) maxBag);
        } else {
            if (solutionType == TREE) {
                if (nextNode == JOIN) {
                    bagSizeNode = std::min((cl_long) (maxMemoryBuffer / s / 2 - node.variables.size() * sizeof(cl_long) * 3), std::min((cl_long) std::min((memorySize - usedMemory - edge1.maxSize * s - edge2.maxSize * s) / s / 2, (memorySize - usedMemory) / 2 / 3 / s), 1l << node.variables.size()));
                } else if (nextNode == INTRODUCEFORGET) {
                    bagSizeNode = std::min((cl_long) (maxMemoryBuffer / s / 2 - node.variables.size() * sizeof(cl_long) * 3), std::min((cl_long) std::min((memorySize - usedMemory - edge1.maxSize * s - edge2.maxSize * s) / s / 2, (memorySize - usedMemory) / 2 / 2 / s), 1l << node.variables.size()));
                }
            } else if (solutionType == ARRAY) {
                bagSizeNode = 1l << (cl_long) std::min(node.variables.size(), (size_t) std::min(log2(maxMemoryBuffer / sizeof(cl_long)), log2(memorySize / sizeof(cl_long) / 3)));
            }
        }

        int64_t maxSize = std::ceil((1l << (node.variables.size())) * 1.0 / bagSizeNode);

        node.solution.clear();

        for (cl_long _a = 0, run = 0; _a < maxSize; _a++, run++) {
            ArraySolution solution;
            solution.minId = run * bagSizeNode;
            solution.maxId = std::min(run * bagSizeNode + bagSizeNode,
                 1l << (node.variables.size()));
            solution.numSolutions = 0;
            std::cout << "run: " << run << " " << solution.minId << " " << solution.maxId << std::endl;
            solution.size = (solution.maxId - solution.minId) + node.variables.size();
            solution.elements = new double[solution.size];
            for (size_t i=0; i < solution.size; i++) {
                solution.elements[i] = -1.0;
            }

            CudaBuffer<double> buf_sol(solution.elements, solution.size);
            CudaBuffer<int64_t> buf_solBag(&(solution.numSolutions), 1);

            for (cl_long b = 0; b < std::max(edge1.solution.size(), edge2.solution.size()); b++) {

                std::unique_ptr<CudaBuffer<int64_t>> buf_sol1( std::make_unique<CudaBuffer<int64_t>>() );
                if (b < edge1.solution.size() && dataPtr(edge1.solution[b]) != NULL) {
                    buf_sol1 = std::make_unique<CudaBuffer<int64_t>>(
                        dataPtr(edge1.solution[b]),
                        dataStructureSize(edge1.solution[b])
                    );
                }

                std::unique_ptr<CudaBuffer<cl_long>> buf_sol2( std::make_unique<CudaBuffer<cl_long>>() );
                if (b < edge2.solution.size() && dataPtr(edge2.solution[b]) != NULL) {
                    buf_sol2 = std::make_unique<CudaBuffer<int64_t>>(
                        dataPtr(edge2.solution[b]),
                        dataStructureSize(edge2.solution[b])
                    );
                }

                long id_offset = solution.minId;
                size_t threads = static_cast<size_t>(solution.maxId - solution.minId);
                std::cerr << "thread offset: " << id_offset << " threads " << threads << std::endl;

                solveJoinWrapper(
                    (int64_t*)buf_sol.device_mem,
                    buf_sol1->device_mem,
                    buf_sol2->device_mem,
                    GPUVars {
                        .count = (int64_t)node.variables.size(),
                        .vars = buf_solVars.device_mem
                    },
                    GPUVars {
                        .count = (int64_t)edge1.variables.size(),
                        .vars = buf_solVars1.device_mem
                    },
                    GPUVars {
                        .count = (int64_t)edge2.variables.size(),
                        .vars = buf_solVars2.device_mem
                    },
                    (b < edge1.solution.size()) ? minId(edge1.solution[b]) : -1,
                    (b < edge1.solution.size()) ? maxId(edge1.solution[b]) : -1,
                    (b < edge2.solution.size()) ? minId(edge2.solution[b]) : -1,
                    (b < edge2.solution.size()) ? maxId(edge2.solution[b]) : -1,
                    solution.minId,
                    (b < edge1.solution.size()) ? minId(edge1.solution[b]) : 0,
                    (b < edge2.solution.size()) ? minId(edge2.solution[b]) : 0,
                    buf_weights->device_mem,
                    buf_solBag.device_mem,
                    pow(2, edge1.exponent + edge2.exponent),
                    buf_exponent.device_mem,
                    threads,
                    id_offset,
                    solve_mode
                );
            }

            buf_solBag.read(&(solution.numSolutions));
            std::cerr << "num solutions (join): " << solution.numSolutions << std::endl;
            std::cout << "a is " << node.solution.size() << std::endl;
            if (solution.numSolutions == 0) {
                delete [] solution.elements;
                solution.elements = NULL;

                if (solutionType == TREE) {
                    if (node.solution.size() > 0) {
                        auto& last = std::get<TreeSolution>(node.solution.back());
                        last.maxId = solution.maxId;
                    } else {
                        node.solution.push_back(std::move(solution));
                    }
                } else {
                    node.solution.push_back(std::move(solution));
                }
            } else {
                // node.elements is an array here
                buf_sol.read(solution.elements);
                this->isSat = 1;
                if (solutionType == TREE) {
                    TreeSolution* last = NULL;
                    if (!node.solution.empty()) {
                        last = &std::get<TreeSolution>(node.solution.back());
                    }

                    // previous bag is not empty, combine if there is still space.
                    if (last != NULL && last->tree != NULL && (solution.numSolutions + last->numSolutions + 2) < solution.size) {
                        std::cerr << "first branch" << std::endl;
                        auto tree = arrayToTree(
                            solution,
                            (bagSizeNode) * 2,
                            node.variables.size(),
                            node,
                            last->numSolutions + 1
                        );
                        combineTree(tree, *last, node.variables.size());
                        tree.size = tree.numSolutions + 1;
                        node.solution.back() = std::move(tree);
                    // previous back is empty, replace it
                    } else if (last != NULL && last->tree == NULL) {
                        std::cerr << "second branch" << std::endl;
                        auto tree = arrayToTree(
                            solution,
                            (bagSizeNode) * 2,
                            node.variables.size(),
                            node,
                            0
                        );
                        tree.minId = last->minId;
                        tree.size = tree.numSolutions + 1;
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
                        tree.size = tree.numSolutions + 1;
                        node.solution.push_back(std::move(tree));
                    }
                    node.maxSize = std::max(node.maxSize,
                        (int64_t)dataStructureSize(node.solution.back()));
                } else if (solutionType == ARRAY) {
                    solution.size = bagSizeNode;
                    node.maxSize = std::max(node.maxSize, (int64_t)solution.size);
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
        std::cerr << "JOIN output hash: " << bagTypeHash(node) << std::endl;
    }

    void Solver::solveIntroduceForget(satformulaType &formula, BagType &pnode, BagType &node, BagType &cnode, bool leaf, nodeType nextNode) {

        std::cerr << "IF input hash: " << std::endl;
        std::cerr << "  " << bagTypeHash(pnode) << std::endl;
        std::cerr << "  " << bagTypeHash(node) << std::endl;
        std::cerr << "  " << bagTypeHash(cnode) << std::endl;
        isSat = 0;
        std::vector<cl_long> fVars;
        std::set_intersection(
                node.variables.begin(), node.variables.end(),
                pnode.variables.begin(), pnode.variables.end(),
        std::back_inserter(fVars));
        std::vector<cl_long> iVars = node.variables;
        std::vector<cl_long> eVars = cnode.variables;

        std::cerr << "variables: " << node.variables.size() << std::endl;
        std::cerr << "fvars: " << fVars.size() << std::endl;

        node.variables = fVars;

        this->numIntroduceForget++;

        // get clauses which only contain iVars
        std::vector<cl_long> numVarsClause;
        std::vector<cl_long> clauses;
        cl_long numClauses = 0;
        for (cl_long i = 0; i < formula.clauses.size(); i++) {
            std::vector<cl_long> v;
            std::set_intersection(iVars.begin(), iVars.end(), formula.clauses[i].begin(), formula.clauses[i].end(), back_inserter(v), compVars);
            if (v.size() == formula.clauses[i].size()) {
                numClauses++;
                numVarsClause.push_back(formula.clauses[i].size());
                for (cl_long a = 0; a < formula.clauses[i].size(); a++) {
                    clauses.push_back(formula.clauses[i][a]);
                }
            }
        }

        node.exponent = CL_LONG_MIN;

        CudaBuffer<cl_long> buf_varsE(eVars);
        CudaBuffer<cl_long> buf_varsI(iVars);
        CudaBuffer<cl_long> buf_clauses(clauses);
        CudaBuffer<cl_long> buf_numVarsC(&numVarsClause[0], numClauses);
        CudaBuffer<cl_double> buf_weights(formula.variableWeights, formula.numWeights);
        CudaBuffer<cl_long> buf_exponent(&(node.exponent), 1);

        size_t usedMemory = sizeof(cl_long) * eVars.size() + sizeof(cl_long) * iVars.size() * 3 + sizeof(cl_long) * (clauses.size()) + sizeof(cl_long) * (numClauses) + sizeof(cl_double) * formula.numWeights + sizeof(cl_long) * fVars.size() + sizeof(cl_double) * formula.numWeights;
        cl_long bagSizeForget = 1;
        cl_long s = sizeof(cl_long);

        if (maxBag > 0) {
            bagSizeForget = 1l << (cl_long) std::min(node.variables.size(), (size_t) maxBag);
        } else {
            if (solutionType == TREE) {
                if (nextNode == JOIN) {
                    bagSizeForget = std::min((cl_long) (maxMemoryBuffer / s / 2 - 3 * node.variables.size() * sizeof(cl_long)), std::min((cl_long) std::min((memorySize - usedMemory - cnode.maxSize * s) / s / 2, (memorySize - usedMemory) / 2 / 3 / s), 1l << node.variables.size()));
                } else if (nextNode == INTRODUCEFORGET) {
                    bagSizeForget = std::min((cl_long) (maxMemoryBuffer / s / 2 - 3 * node.variables.size() * sizeof(cl_long)), std::min((cl_long) std::min((memorySize - usedMemory - cnode.maxSize * s) / s / 2, (memorySize - usedMemory) / 2 / 2 / s), 1l << node.variables.size()));
                }
            } else if (solutionType == ARRAY) {
                bagSizeForget = 1l << (cl_long) std::min(node.variables.size(), (size_t) std::min(log2(maxMemoryBuffer / sizeof(cl_long)), log2(memorySize / sizeof(cl_long) / 3)));
            }
        }

        cl_long maxSize = std::ceil((1l << (node.variables.size())) * 1.0 / bagSizeForget);
        std::cerr << "bag size forget: " << bagSizeForget << std::endl;
        std::cerr << "variables: " << node.variables.size() << std::endl;
        std::cerr << "used memory: " << usedMemory << std::endl;

        node.solution.clear();

        for (cl_long _a = 0, run = 0; _a < maxSize; _a++, run++) {

            int64_t sol_minId =  run * bagSizeForget;
            int64_t sol_maxId = std::min(run * bagSizeForget + bagSizeForget, 1l << (node.variables.size()));

            std::variant<TreeSolution, ArraySolution> solution = initializeSolution(
                solutionType,
                sol_minId,
                sol_maxId,
                (sol_maxId - sol_minId) * 2 + node.variables.size()
            );

            CudaBuffer<cl_long> buf_solsF(dataPtr(solution), dataStructureSize(solution));
            CudaBuffer<cl_long> buf_varsF(fVars);
            CudaBuffer<cl_long> buf_solBag(numSolutionsPtr(solution), 1);

            for (auto &csol : cnode.solution) {
                if (dataPtr(csol) == NULL) {
                    continue;
                }

                std::unique_ptr<CudaBuffer<int64_t>> buf_solsE( std::make_unique<CudaBuffer<int64_t>>() );
                if (!leaf) {
                    buf_solsE = std::make_unique<CudaBuffer<cl_long>>(dataPtr(csol), dataStructureSize(csol));
                }

                size_t threads = static_cast<size_t>(maxId(solution) - minId(solution));
                long id_offset = minId(solution);

                int64_t combinations = (int64_t) pow(2, iVars.size() - fVars.size());

                // FIXME: offset onto global id
                introduceForgetWrapper(
                    buf_solsF.device_mem,
                    GPUVars {
                        .count = (int64_t)fVars.size(),
                        .vars = buf_varsF.device_mem
                    },
                    buf_solsE->device_mem,
                    GPUVars {
                        .count = (int64_t)eVars.size(),
                        .vars = buf_varsE.device_mem
                    },
                    combinations,
                    minId(csol),
                    maxId(csol),
                    minId(solution),
                    minId(csol),
                    buf_solBag.device_mem,
                    GPUVars {
                        .count = (int64_t)buf_varsI.size(),
                        .vars = buf_varsI.device_mem
                    },
                    buf_clauses.device_mem,
                    buf_numVarsC.device_mem,
                    numClauses,
                    buf_weights.device_mem,
                    buf_exponent.device_mem,
                    pow(2, cnode.exponent),
                    threads,
                    id_offset,
                    solve_mode
                );
            }
            buf_solBag.read(numSolutionsPtr(solution));

            std::cerr << "num solutions: " << numSolutions(solution) << std::endl;
            if (numSolutions(solution) == 0) {
                freeData(solution);

                if (solutionType == TREE) {
                    if (node.solution.size() > 0) {
                        auto& last = std::get<TreeSolution>(node.solution.back());
                        last.maxId = maxId(solution);
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
                        sol.numSolutions--;
                    }
                    delete [] sol.tree;
                    sol.tree = NULL;
                    if (last != NULL && last->tree != NULL
                        && (sol.numSolutions + last->numSolutions + 2) < sol.size) {

                        int64_t nodeCount = sol.numSolutions + last->numSolutions + 2;
                        sol.tree = new TreeNode[nodeCount];
                        cudaMemcpy(sol.tree, buf_solsF.device_mem, sizeof(int64_t) * nodeCount, cudaMemcpyDeviceToHost);

                        combineTree(sol, *last, node.variables.size());
                        sol.size = sol.numSolutions + 1;
                        node.solution.back() = std::move(sol);
                    } else {
                        int64_t nodeCount = sol.numSolutions + 1;
                        sol.tree = new TreeNode[nodeCount];
                        sol.size = sol.numSolutions + 1;

                        cudaMemcpy(sol.tree, buf_solsF.device_mem, sizeof(int64_t) * nodeCount, cudaMemcpyDeviceToHost);

                        // replace previous bag if needed
                        if (last != NULL && last->tree == NULL) {
                            sol.minId = last->minId;
                            node.solution.back() = std::move(sol);
                        } else {
                            node.solution.push_back(std::move(sol));
                        }
                    }
                    node.maxSize = std::max(node.maxSize,
                        (int64_t)dataStructureSize(node.solution.back()));
                } else if (solutionType == ARRAY) {
                    auto& sol = std::get<ArraySolution>(solution);
                    sol.size = bagSizeForget;

                    cudaMemcpy(sol.elements, buf_solsF.device_mem, sizeof(int64_t) * bagSizeForget, cudaMemcpyDeviceToHost);
                    node.solution.push_back(std::move(solution));
                }
            }
        }

        buf_exponent.read(&(node.exponent));
        std::cerr << "exponent: " << node.exponent << std::endl;
        node.correction = cnode.correction + cnode.exponent;
        cl_long tableSize = 0;
        for (const auto &sol : node.solution) {
            tableSize += dataStructureSize(sol);
        }
        std::cout << "table size: " << tableSize << std::endl;
        std::cerr << "IF output hash: " << bagTypeHash(node) << std::endl;

        this->maxTableSize = std::max(this->maxTableSize, tableSize);
        for (auto &csol : cnode.solution) {
            freeData(csol);
        }
    }
}
