#define GPU_HOST_ATTR __device__ __host__

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>

#include <gpusat_types.h>
#include "kernel.h"


// TODO: get rid of RunMeta

namespace gpusat {


__device__ int64_t get_global_id(uint64_t minId, uint64_t maxId) {
    // TODO: y and z
    int64_t id = blockDim.x * blockIdx.x + threadIdx.x + minId;
    if (id >= maxId) {
        return -1;
    }
    return id;
}

/**
 * combines two tree structure into one
 *
 * @param numVars
 *      the number of variables in the bag
 * @param tree
 *      a pointer to the tree structure which will receive all the models from the other tree
 * @param solutions_old
 *      a pointer to the old tree structure
 * @param treeSize
 *      the number of nodes in the tree
 * @param startId
  *     the start id of the current node
 */
__global__ void combineTree(
        TreeSolution<GpuOnly>* to,
        const TreeSolution<GpuOnly>* from
) {
    // this is is relative to the <from>-tree
    int64_t id = get_global_id(from->minId(), from->maxId());
    if (id < 0) {
        return;
    }
    double val = from->solutionCountFor(id);
    if (val > 0) {
        to->setCount(id, val);
    }
}

/**
 * Operation to solve a Introduce node in the decomposition.
 *
 * @param variables
 *      the ids of the variables in the current bag
 * @param edge
 *      the number of models for each assignment of the next bag
 * @param edgeVariables
 *      variables in the next bag
 * @param minId
 *      the start id of the last bag
 * @param maxId
 *      the end id of the last bag
 * @param weights
 *      the variables weights for weighted model counting
 * @param id
 *      the id for which the introduce should be solved
 * @return
 *      the model count
 */
template<class T>
__device__ double solveIntroduce_(
        GPUVars variables,
        const T& edge,
        GPUVars edgeVariables,
        double *weights,
        int64_t id
) {
    int64_t otherId = 0;
    int64_t a = 0, b = 0;
    double weight = 1.0;
    for (b = 0; b < edgeVariables.count && a < variables.count; b++) {
        while ((variables.vars[a] != edgeVariables.vars[b])) {
            a++;
        }

        otherId = otherId | (((id >> a) & 1) << b);
        a++;
    };

    //weighted model count
    if (weights != 0) {
        for (b = 0, a = 0; a < variables.count; a++) {
            if (edgeVariables.vars == 0 || (variables.vars[a] != edgeVariables.vars[b])) {
                weight *= weights[((id >> a) & 1) > 0 ? variables.vars[a] * 2 : variables.vars[a] * 2 + 1];
            }
            if (edgeVariables.vars != 0 && (variables.vars[a] == edgeVariables.vars[b]) && (b < (edgeVariables.count - 1))) {
                b++;
            }
        }
    }

    if (edge.hasData() && otherId >= edge.minId() && otherId < edge.maxId()) {
        return max(edge.solutionCountFor(otherId) * weight, 0.0); 
    } else if (!edge.hasData() && otherId >= edge.minId() && otherId < edge.maxId()) {
        return 0.0;
    } else {
        // prevent inclusion of this branch
        return -1.0;
    }
}

/**
 * Operation to check if an assignment satisfies the clauses of a SAT formula.
 *
 * @param clauses
 *      the clauses in the SAT formula
 * @param numVarsC
 *      array containing the number of Variables in each clause
 * @param numclauses
 *      the number of clauses in the sat formula
 * @param id
 *      the id of the thread - used to get the variable assignment
 * @param variables
 *      a vector containing the ids of the variables
 * @return
 *      1 - if the assignment satisfies the formula
 *      0 - if the assignment doesn't satisfy the formula
 */
__device__ int checkBag(uint64_t *clauses, long numClauses, int64_t id, GPUVars variables) {
    uint64_t unsigned_id = abs(id);
    for (long i = 0; i < numClauses; i++) {
        uint64_t c_vars = clauses[2*i];
        uint64_t c_neg = clauses[2*i+1];
        if (((unsigned_id^c_neg) & c_vars) == 0) {
            return 0;
        }
    }
    return 1;
}

/**
 * Operation to solve a Join node in the decomposition.
 *
 * @param solutions
 *      the number of solutions of the join
 * @param edge1
 *      contains the number of solutions in the first edge
 * @param edge2
 *      contains the number of solutions in the second edge
 * @param variables
 *      the variables in the join bag
 * @param edgeVariables1
 *      the variables in the bag of the first edge
 * @param edgeVariables2
 *      the variables in the bag of the second edge
 * @param numV
 *      the number of variables in the join bag
  * @param minId1
  *     the start id of the first edge
  * @param maxId1
  *     the end id of the first edge
  * @param minId2
  *     the start id of the second edge
  * @param maxId2
  *     the end id of the second edge
  * @param startIDNode
  *     the start id of the current node
  * @param weights
  *     the variable weights for weighted model counting
  * @param sols
  *     the number of assignments which lead to a solution
  * @param value
  *     correction value for the exponents
  * @param exponent
  *     the max exponent of this run
  */

template <class T, class E1, class E2>
__global__ void solveJoin(
        T* solution,
        E1* edge1,
        E2* edge2,
        GPUVars variables,
        GPUVars edgeVariables1,
        GPUVars edgeVariables2,
        //int64_t startIDNode,
        //int64_t startIDEdge1,
        //int64_t startIDEdge2,
        double *weights,
        //uint64_t *sols,
        double value,
        int32_t *exponent,
        const SolveConfig cfg
) {
    int64_t id = get_global_id(solution->minId(), solution->maxId());
    if (id < 0) {
        return;
    }

    double edge1_solutions = -1.0;
    double edge2_solutions = -1.0;
    double weight = 1.0;

    if (edge1 != nullptr) {
        edge1_solutions = solveIntroduce_(variables, *edge1, edgeVariables1, weights, id);
    }
    if (edge2 != nullptr) {
        edge2_solutions = solveIntroduce_(variables, *edge2, edgeVariables2, weights, id);
    } 

    // weighted model count
    if (weights != 0) {
        for (long a = 0; a < variables.count; a++) {
            weight *= weights[((id >> a) & 1) > 0 ? variables.vars[a] * 2 : variables.vars[a] * 2 + 1];
        }
    }

    double solution_value = -1.0;
    // indicates that this value is incomplete
    // and should not yet be used to derive a possible exponent
    bool value_incomplete = false;

    // do both edges have an entry for edge1 and edge2?
    if (edge1_solutions >= 0.0 && edge2_solutions >= 0.0) {
        // only store a solution if both edges have a count > 0
        if (edge1_solutions > 0.0 && edge2_solutions > 0.0) {
            solution->setSatisfiability(true);

            // we do not need to consider the old solution
            // count in this bag, because each id can only occur
            // in an edge node once, and here the id occurs for both edges.

            solution_value = edge1_solutions * edge2_solutions / weight;
            //atomicAdd(sols, 1);
            if (!cfg.no_exponent) {
                solution_value /= value;
            }
        }
    // we need to consider individual edges and maybe look
    // at we have already stored for the current id.
    } else if (edge1_solutions >= 0.0 || edge2_solutions >= 0.0) {
        double oldVal = solution->solutionCountFor(id); 

        // if the solution was not present before, multiply with one.
        if (oldVal < 0.0) {
            value_incomplete = true;
            oldVal = 1.0;
        }

        // In order to not use weight and value twice,
        // move them to different branches of the calculation.
        // If one edge is 0, the whole will be 0 anyway.

        // edge 1 has the solution
        if (edge1_solutions > 0.0) {
            // use weight here
            solution_value = edge1_solutions * oldVal / weight;
        } else if (edge2_solutions > 0.0) {
            solution_value = edge2_solutions * oldVal;
            // use value here
            if (!cfg.no_exponent) {
                solution_value /= value;
            }
        } else {
            solution_value = 0.0;
        }
    }

    // if we found solutions, store them
    if (solution_value >= 0.0) {
        solution->setCount(id, solution_value);
        if (!value_incomplete) {
            solution->setSatisfiability(true);
            if (!cfg.no_exponent) {
                atomicMax(exponent, ilogb(solution_value));
            }
        }
    }
}

/**
 * Operation to solve an Introduce
 *
 * @param clauses
 *      the clauses in the sat formula
 * @param numVarsC
 *      the number of variables for each clause
 * @param numclauses
 *      the number of clauses
 * @param variables
 *      the ids of the variables in the current bag
 * @param edge
 *      the number of models for each assignment of the last bags
 * @param edgeVariables
 *      variables of the last bag
 * @param minId
 *      the start id of the last bag
 * @param maxId
 *      the end id of the last bag
 * @param weights
 *      the variables weights for weighted model counting
 * @param id
 *      the id for which the introduce should be solved
 * @return
 *      the model count
 */

template <class T>
__device__ double solveIntroduceF(
        uint64_t *clauses,
        long numclauses,
        GPUVars variables,
        const T* edge,
        GPUVars edgeVariables,
        double *weights,
        long id
) {
    double tmp;
    if (edge != nullptr && edge->hasData()) {
        // get solutions count edge
        tmp = solveIntroduce_(variables, *edge, edgeVariables, weights, id);
    } else {
        // no edge - solve leaf
        tmp = 1.0;

        //weighted model count
        if (weights != 0) {
            for (long i = 0; i < variables.count; i++) {
                tmp *= weights[((id >> i) & 1) > 0 ? variables.vars[i] * 2 : variables.vars[i] * 2 + 1];
            }
        }
    }
    if (tmp > 0.0) {
        // check if assignment satisfies the given clauses
        int sat = checkBag(clauses, numclauses, id, variables);
        if (sat != 1) {
            return 0.0;
        } else {
            return tmp;
        }
    } else {
        return 0.0;
    }
}

/**
 * Operation to solve a Introduce and Forget node in the decomposition.
 *
 * @param solsF
 *      the number of models for each assignment
 * @param varsForget
 *      the variables after the forget operation
 * @param solsE
 *      the solutions from the last node
 * @param lastVars
 *      the variables from the alst oepration
 * @param combinations
 *      the number of assignments for which we have to collect the model counts
 * @param minIdE
 *      start id of the chunk from the last node
 * @param maxIdE
 *      end id of the chunk from the last node
 * @param startIDF
 *      start id of the chung from the current node
 * @param sols
  *     the number of assignments which lead to a solution
 * @param varsI
 *      the variables after the introduce
 * @param clauses
 *      the clauses which only contain variables from the introduce operation
 * @param numclauses
 *      the number of clauses
 * @param weights
 *      the variables weights for weighted model counting
 * @param exponent
  *     the max exponent of this run
 * @param value
  *     correction value for the exponents
 */
template <class T, class E>
__global__ void solveIntroduceForget(
        T* solsF,
        GPUVars varsForget,
        const E* solsE,
        GPUVars lastVars,
        uint64_t combinations,
        GPUVars varsIntroduce,
        uint64_t *clauses,
        long numclauses,
        double *weights,
        int32_t *exponent,
        double value,
        const SolveConfig cfg
) {
    int64_t id = get_global_id(solsF->minId(), solsF->maxId());
    if (id < 0) {
        return;
    }

    if (varsIntroduce.count != varsForget.count) {
        double tmp = 0;
        long templateId = 0;
        // generate templateId
        for (long i = 0, a = 0; i < varsIntroduce.count && a < varsForget.count; i++) {
            if (varsIntroduce.vars[i] == varsForget.vars[a]) {
                templateId = templateId | (((id >> a) & 1) << i);
                a++;
            }
        }

        // iterate though all corresponding edge solutions
        for (long i = 0; i < combinations; i++) {
            long b = 0, otherId = templateId;
            for (long a = 0; a < varsIntroduce.count; a++) {
                if (b >= varsForget.count || varsIntroduce.vars[a] != varsForget.vars[b]) {
                    otherId = otherId | (((i >> (a - b)) & 1) << a);
                } else {
                    b++;
                }
            }
            tmp += solveIntroduceF(clauses, numclauses, varsIntroduce, solsE, lastVars, weights, otherId);
        }
        
        if (tmp > 0) {
            double last = solsF->solutionCountFor(id);
            last = max(last, 0.0);
            if (!cfg.no_exponent)  {
                solsF->setCount(id, (tmp / value + last));
                atomicMax(exponent, ilogb((tmp / value + last)));
                assert(*exponent != FP_ILOGB0);
            } else {
                solsF->setCount(id, (tmp + last));
            }
            solsF->setSatisfiability(true);
        }
    } else {
        // no forget variables, only introduce
        double tmp = solveIntroduceF(clauses, numclauses, varsIntroduce, solsE, lastVars, weights, id);
        if (tmp > 0) {
            double last = solsF->solutionCountFor(id);
            last = max(last, 0.0);
            if (!cfg.no_exponent)  {
                solsF->setCount(id, (tmp / value + last));
                atomicMax(exponent, ilogb((tmp / value + last)));
                assert(*exponent != FP_ILOGB0);
            } else {
                solsF->setCount(id, (tmp + last));
            }
            solsF->setSatisfiability(true);
        }
    } 
}

/**
 * Clones a solution bag to the GPU,
 * this clone uses the data of the <CudaMem> bag it was cloned from
 * and does not own it.
 */

template <template<typename> typename T>
std::unique_ptr<T<GpuOnly>, CudaMem> gpuClone(const T<CudaMem>& owner) {
    static_assert(sizeof(T<CudaMem>) == sizeof(T<GpuOnly>));
    
    // Note that since clone is malloc'ed here,
    // its destructor will not be called. 
    T<GpuOnly>* clone = nullptr;
    gpuErrchk(cudaMalloc(&clone, sizeof(T<GpuOnly>)));
    assert(clone != nullptr);
    gpuErrchk(cudaMemcpy(clone, &owner, sizeof(T<GpuOnly>), cudaMemcpyHostToDevice));
    return std::unique_ptr<T<GpuOnly>, CudaMem>(clone);
}

TreeSolution<CudaMem> combineTreeWrapper(
    TreeSolution<CudaMem>& to_owner,
    const TreeSolution<CudaMem>& from_owner
) {
    auto to_gpu = gpuClone(to_owner);
    auto from_gpu = gpuClone(from_owner);

    int64_t threadsPerBlock = 512;
    int64_t threads = from_owner.maxId() - from_owner.minId();
    int64_t blocksPerGrid = (threads + threadsPerBlock - 1) / threadsPerBlock;
    combineTree<<<blocksPerGrid, threadsPerBlock>>>(to_gpu.get(), from_gpu.get());
    gpuErrchk(cudaDeviceSynchronize());
    update(to_owner, to_gpu);
    return std::move(to_owner);
}

void solveJoinWrapper( 
    CudaSolutionVariant& solution_variant,
    const std::optional<CudaSolutionVariant>& edge1,
    const std::optional<CudaSolutionVariant>& edge2,
    GPUVars variables,
    GPUVars edgeVariables1,
    GPUVars edgeVariables2,
    double *weights,
    double value,
    int32_t *exponent,
    const SolveConfig cfg
) {
    int64_t threadsPerBlock = 512;
    int64_t threads = maxId(solution_variant) - minId(solution_variant);
    int64_t blocksPerGrid = (threads + threadsPerBlock - 1) / threadsPerBlock;

    assert(edge1.has_value() || edge2.has_value());

    std::visit([&](auto& solution) {
        auto solution_gpu = gpuClone(solution);

        auto single_edge1_join = [&](const CudaSolutionVariant& edge) {
            std::visit([&](const auto &edge_owner) {
                auto edge_gpu = gpuClone(edge_owner);

                solveJoin<<<blocksPerGrid, threadsPerBlock>>>(
                    solution_gpu.get(),
                    edge_gpu.get(),
                    (decltype(edge_gpu.get()))nullptr,
                    variables,
                    edgeVariables1,
                    edgeVariables2,
                    weights,
                    value, 
                    exponent,
                    cfg
                );
                gpuErrchk(cudaDeviceSynchronize());
            }, edge);
        };

        auto single_edge2_join = [&](const CudaSolutionVariant& edge) {
            std::visit([&](const auto &edge_owner) {
                auto edge_gpu = gpuClone(edge_owner);

                solveJoin<<<blocksPerGrid, threadsPerBlock>>>(
                    solution_gpu.get(),
                    (decltype(edge_gpu.get()))nullptr,
                    edge_gpu.get(),
                    variables,
                    edgeVariables1,
                    edgeVariables2,
                    weights,
                    value, 
                    exponent,
                    cfg
                );
                gpuErrchk(cudaDeviceSynchronize());
            }, edge);
        };

        auto double_edge_join = [&](const CudaSolutionVariant& e1, const CudaSolutionVariant& e2) {
            std::visit([&](const auto &e1_owner) {
                auto e1_gpu = gpuClone(e1_owner);

                std::visit([&](const auto &e2_owner) {
                    auto e2_gpu = gpuClone(e2_owner);

                    solveJoin<<<blocksPerGrid, threadsPerBlock>>>(
                        solution_gpu.get(),
                        e1_gpu.get(),
                        e2_gpu.get(),
                        variables,
                        edgeVariables1,
                        edgeVariables2,
                        weights,
                        value, 
                        exponent,
                        cfg
                    );
                    gpuErrchk(cudaDeviceSynchronize());
                }, e2);
            }, e1);
        };

        if (edge1.has_value() && !edge2.has_value()) {
            single_edge1_join(edge1.value());
        } else if (!edge1.has_value() && edge2.has_value()) {
            single_edge2_join(edge2.value());
        } else {
            assert(edge1.has_value() && edge2.has_value());
            double_edge_join(edge1.value(), edge2.value());
        }
        update(solution, solution_gpu);
    }, solution_variant);
}

void introduceForgetWrapper(
    CudaSolutionVariant& solution_owner,
    GPUVars varsForget,
    const std::optional<CudaSolutionVariant>& edge,
    GPUVars lastVars,
    GPUVars varsIntroduce,
    // FIXME: Move this static information to GPU once.
    uint64_t *clauses,
    long numclauses,
    double *weights,
    int32_t *exponent,
    double previous_value,
    const SolveConfig cfg
) {
    int64_t threadsPerBlock = 512;
    int64_t threads = maxId(solution_owner) - minId(solution_owner);
    int64_t blocksPerGrid = (threads + threadsPerBlock - 1) / threadsPerBlock;
    uint64_t combinations = (uint64_t) pow(2, varsIntroduce.count - varsForget.count);

    std::visit([&](auto& solsF) {
        auto solution_gpu = gpuClone(solsF);
        
        // leaf node
        if (!edge.has_value()) {
            solveIntroduceForget<<<blocksPerGrid, threadsPerBlock>>>(
                solution_gpu.get(),
                varsForget,
                (decltype(solution_gpu.get()))nullptr,
                lastVars,
                combinations,
                varsIntroduce,
                clauses,
                numclauses,
                weights,
                exponent, 
                previous_value,
                cfg
            );
            gpuErrchk(cudaDeviceSynchronize());
        } else {
            std::visit([&](const auto& solsE) {
                auto edge_gpu = gpuClone(solsE);
 
                solveIntroduceForget<<<blocksPerGrid, threadsPerBlock>>>(
                    solution_gpu.get(),
                    varsForget,
                    edge_gpu.get(),
                    lastVars,
                    combinations,
                    varsIntroduce,
                    clauses,
                    numclauses,
                    weights,
                    exponent, 
                    previous_value,
                    cfg
                );
                gpuErrchk(cudaDeviceSynchronize());
            }, edge.value());
        }
        update(solsF, solution_gpu);
    }, solution_owner);
}


// memory initialization for the ArraySolution
void meminit(ArraySolution<CudaMem>& sol, size_t from, size_t to) { 
    auto start = thrust::device_ptr<decltype(sol.initializer())>(sol.data());
    thrust::fill(start + from, start + to, sol.initializer());
}

// memory initialization for the TreeSolution
void meminit(TreeSolution<CudaMem>& sol, size_t from, size_t to) { 
    // make sure that we can simplyfy to a memset
    assert(sol.initializer().empty == 0);
    static_assert(std::is_same<decltype(sol.initializer().empty), uint64_t>::value);

    gpuErrchk(cudaMemset(sol.data() + from, 0, (to - from) * sol.elementSize()));
}

    /**
     * Returns a solution bag that lives on the CPU
     * but owns GPU data copied from the input bag.
     */
    template <template<typename> typename T>
    T<CudaMem> gpuOwner(const T<CpuMem>& orig, size_t reserve) {
        // copy parameters
        T<CudaMem> gpu(orig, nullptr, 0);

        gpu.setDataStructureSize(gpu.dataStructureSize() + reserve);

        // allocate GPU memory
        gpu.allocate();

        if (orig.hasData()) {
            // copy data structure
            gpuErrchk(cudaMemcpy(
                gpu.data(),
                orig.data(),
                orig.dataStructureSize() * orig.elementSize(),
                cudaMemcpyHostToDevice
            ));
            // reserve additional elements if desired
            if (reserve) {
                meminit(gpu, orig.dataStructureSize(), gpu.dataStructureSize());
            }
        } else {
            meminit(gpu, 0, gpu.dataStructureSize());
        }
        return gpu;
    }

    // Explicitly instantiate for all datastructures
    // To ensure the compiler does not optimize them out.
    template ArraySolution<CudaMem> gpuOwner(const ArraySolution<CpuMem>&, size_t);
    template TreeSolution<CudaMem> gpuOwner(const TreeSolution<CpuMem>&, size_t);

    CudaSolutionVariant gpuOwner(const SolutionVariant& orig, size_t reserve) {
        return std::visit([&](auto& sol) -> CudaSolutionVariant {
            return std::variant<TreeSolution<CudaMem>, ArraySolution<CudaMem>>(std::move(gpuOwner(sol, reserve)));
        }, orig);
    }


    /**
     * Returns a solution bag that is constructed
     * (copied) from a bag tat owns GPU data.
     */
    template <template<typename> typename T>
    T<CpuMem> cpuCopy(const T<CudaMem>& gpu, size_t reserve) {
        // copy parameters
        T<CpuMem> cpu(gpu, nullptr, 0);

        cpu.setDataStructureSize(cpu.dataStructureSize() + reserve);

        // allocate CPU memory
        cpu.allocate();

        gpuErrchk(cudaDeviceSynchronize());
        assert(gpu.hasData());
        // copy data structure
        gpuErrchk(cudaMemcpy(
            cpu.data(),
            gpu.data(),
            gpu.dataStructureSize() * gpu.elementSize(),
            cudaMemcpyDeviceToHost
        ));

        // reserve additional elements if desired
        if (reserve) {
            std::fill(
                cpu.data() + gpu.dataStructureSize(),
                cpu.data() + cpu.dataStructureSize(),
                cpu.initializer()
            );
        }
        return cpu;
    }

    // Explicitly instantiate for all datastructures
    // To ensure the compiler does not optimize them out.
    template ArraySolution<CpuMem> cpuCopy(const ArraySolution<CudaMem>&, size_t);
    template TreeSolution<CpuMem> cpuCopy(const TreeSolution<CudaMem>&, size_t);

    SolutionVariant cpuCopy(const CudaSolutionVariant& gpu, size_t reserve) {
        return std::visit([&](const auto& gpu_sol) -> SolutionVariant {
            return SolutionVariant(std::move(cpuCopy(gpu_sol, reserve)));
        }, gpu);
    }

}
