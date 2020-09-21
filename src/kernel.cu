#define GPU_HOST_ATTR __device__ __host__

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>

#include "types.h"
#include "kernel.h"


// TODO: get rid of RunMeta

namespace gpusat {


__device__ uint64_t atomicAdd(uint64_t* address, uint64_t val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        (val + (uint64_t)(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return old;
}

__device__ uint64_t atomicSub(uint64_t* address, uint64_t val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        ((uint64_t)(assumed) - val));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return old;
}

__device__ int64_t atomicMax(int64_t* address, int64_t val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        (max(val, (int64_t)(assumed))));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return old;
}

// FIXME: normal atomicAdd might not be atomic across devices


__device__ int64_t get_global_id(uint64_t minId, uint64_t maxId) {
    // TODO: y and z
    int64_t id = blockDim.x * blockIdx.x + threadIdx.x + minId;
    if (id >= maxId) {
        return -1;
    }
    return id;
}

/**
 * returns the model count which corresponds to the given id
 *
 * @param id
 *      the id for which the model count should be returned
 * @param tree
 *      a pointer to the tree structure
 * @param numVars
 *      the number of variables in the bag
 * @return
 *      the model count
 */
/*__host__ __device__ double getCount(int64_t id, const TreeNode *tree, long numVars) {
    moved to TreeSolution class 
}
*/

/**
 * sets the model count which corresponds to the given id
 *
 * @param id
 *      the id for which the model count should be set
 * @param tree
 *      a pointer to the tree structure
 * @param numVars
 *      the number of variables in the bag
 * @param treeSize
 *      the number of nodes in the tree
 * @param value
 *      the new value of the id
 */
/*
__device__ void setCount(uint64_t id, TreeNode *tree, size_t numVars, uint64_t* treeSize, double value) {
    ulong nextId = 0;
    ulong val = 0;
    if (numVars == 0) {
        atomicAdd(treeSize, 1);
    }
    for (ulong i = 0; i < numVars; i++) { 
        // lower or upper 32bit, depending on if bit of variable i is set in id
        uint * lowVal = &((uint *) &(tree[nextId]))[(id >> (numVars - i - 1)) & 1];
        // secure our slot by incrementing treeSize
        if (val == 0 && *lowVal == 0) {
            val = atomicAdd(treeSize, 1) + 1;
        }
        atomicCAS(lowVal, 0, val);
        if (*lowVal == val) {
            if (i < (numVars - 1)) {
                val = atomicAdd(treeSize, 1) + 1;
            }
        }
        nextId = *lowVal;
    }
    tree[nextId].content = value;
}
*/

/**
 * converts a array structure into a tree
 *
 * @param numVars
 *      the number of variables in the bag
 * @param tree
 *      a pointer to the tree structure
 * @param solutions_old
 *      array containing the models
 * @param treeSize
 *      the number of nodes in the tree
 * @param startId
  *     the start id of the current node
 * @param exponent
  *     the max exponent of this run
 */
__global__ void array2tree(
        TreeSolution<GpuOnly>* tree,
        const ArraySolution<GpuOnly>* array,
        int64_t *exponent,
        const RunMeta meta
) {
    assert(tree != nullptr);
    assert(array != nullptr);
    assert(tree->minId() == array->minId());
    assert(tree->maxId() == array->maxId());
    int64_t id = get_global_id(array->minId(), array->maxId());
    if (id < 0) {
        return;
    }
    auto solutions = array->solutionCountFor(id);
    if (solutions > 0) {
        tree->setCount(id, solutions);
        if (!(meta.mode & NO_EXP)) {
            atomicMax(exponent, (int64_t)ilogb(solutions));
        }
    }
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
        const TreeSolution<GpuOnly>* from,
        const RunMeta meta
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
        int64_t id,
        SolveMode mode
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
__device__ int checkBag(long *clauses, long *numVarsC, long numclauses, int64_t id, GPUVars variables) {
    long i, varNum = 0;
    long satC = 0, a, b;
    // iterate through all clauses
    for (i = 0; i < numclauses; i++) {
        satC = 0;
        // iterate through clause variables
        for (a = 0; a < numVarsC[i] && !satC; a++) {
            satC = 1;
            //check current variables
            for (b = 0; b < variables.count; b++) {
                // check if clause is satisfied
                if ((clauses[varNum + a] == variables.vars[b]) ||
                    (clauses[varNum + a] == -variables.vars[b])) {
                    satC = 0;
                    if (clauses[varNum + a] < 0) {
                        //clause contains negative var and var is assigned negative
                        if ((id & (1 << (b))) == 0) {
                            satC = 1;
                            break;
                        }
                    } else {
                        //clause contains positive var and var is assigned positive
                        if ((id & (1 << (b))) > 0) {
                            satC = 1;
                            break;
                        }
                    }
                }
            }
        }
        varNum += numVarsC[i];
        // we have an unsattisifed clause
        if (!satC) {
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
        int64_t *exponent,
        const RunMeta run
) {
    int64_t id = get_global_id(solution->minId(), solution->maxId());
    if (id < 0) {
        return;
    }

    double tmp = -1, tmp_ = -1;
    double weight = 1;
    if (edge1 != nullptr) {
        // get solution count from first edge
        tmp = solveIntroduce_(variables, *edge1, edgeVariables1, weights, id, run.mode);
    }
    if (edge2 != nullptr) {
        // get solution count from second edge
        tmp_ = solveIntroduce_(variables, *edge2, edgeVariables2, weights, id, run.mode);
    }
    // weighted model count
    if (weights != 0) {
        for (long a = 0; a < variables.count; a++) {
            weight *= weights[((id >> a) & 1) > 0 ? variables.vars[a] * 2 : variables.vars[a] * 2 + 1];
        }
    }

    double solution_value = -1.0;
    if (tmp_ >= 0.0 && tmp >= 0.0) {
        if (tmp_ > 0.0 && tmp > 0.0) {
            solution->incSolutions();

            //atomicAdd(sols, 1);
            if (!(run.mode & NO_EXP)) {
                solution_value = tmp_ * tmp / value / weight;
                //solutions[id - startIDNode] = ;
            } else {
                solution_value = tmp_ * tmp / weight;
                //solutions[id - startIDNode] = tmp_ * tmp / weight;
            }
        }
    }

        // we have some solutions in edge1
    else if (tmp >= 0.0) {
        double oldVal = solution->solutionCountFor(id); //solutions[id - startIDNode];
        if (oldVal < 0) {
            if (tmp > 0) {
                solution->incSolutions();
                //atomicAdd(sols, 1);
            }
        } else if (oldVal > 0) {
            if (tmp == 0) {
                solution->decSolutions();
                //atomicSub(sols, 1);
            }
        }
        if (oldVal < 0) {
            oldVal = 1.0;
        }
        // FIXME: Why not respect value / exponent ?
        solution_value = tmp * oldVal / weight;
        //solutions[id - startIDNode] = tmp * oldVal / weight;
    }

        // we have some solutions in edge2
    else if (tmp_ >= 0.0) {
        double oldVal = solution->solutionCountFor(id); //solutions[id - startIDNode];
        if (oldVal < 0) {
            if (tmp_ > 0) {
                solution->incSolutions();
                //atomicAdd(sols, 1);
            }
        } else if (oldVal > 0) {
            if (tmp_ == 0) {
                solution->decSolutions();
                //atomicSub(sols, 1);
            }
        }
        if (oldVal < 0) {
            oldVal = 1.0;
        }

        if (!(run.mode & NO_EXP)) {
            solution_value = tmp_ * oldVal / value;
            //solutions[id - startIDNode] = tmp_ * oldVal / value;
        } else {
            solution_value = tmp_ * oldVal;
            //solutions[id - startIDNode] = tmp_ * oldVal;
        }
    }
    if (solution_value > 0.0) {
        solution->setCount(id, solution_value);
        if (!(run.mode & NO_EXP)) {
            atomicMax(exponent, ilogb(solution_value)); //solutions[id - startIDNode]));
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
        long *clauses,
        long *numVarsC,
        long numclauses,
        GPUVars variables,
        const T* edge,
        GPUVars edgeVariables,
        double *weights,
        long id,
        SolveMode mode
) {
    double tmp;
    if (edge != nullptr && edge->hasData()) {
        // get solutions count edge
        tmp = solveIntroduce_(variables, *edge, edgeVariables, weights, id, mode);
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
        int sat = checkBag(clauses, numVarsC, numclauses, id, variables);
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
 * @param numVarsC
 *      the number of variables per clause
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
        long *clauses,
        long *numVarsC,
        long numclauses,
        double *weights,
        long *exponent,
        double value,
        const RunMeta run
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
            tmp += solveIntroduceF(clauses, numVarsC, numclauses, varsIntroduce, solsE, lastVars, weights, otherId, run.mode);
        }
        
        if (tmp > 0) {
            double last = solsF->solutionCountFor(id);
            last = max(last, 0.0);
            if (!(run.mode & NO_EXP))  {
                solsF->setCount(id, (tmp / value + last));
                atomicMax(exponent, ilogb((tmp / value + last)));
            } else {
                solsF->setCount(id, (tmp + last));
            }
            //if (auto array = std::get_if<ArraySolution>(&solsF)) {
            // count solution (no-op except for ArraySolution)
            solsF->incSolutions();
            //}
        }
    } else {
        // no forget variables, only introduce
        double tmp = solveIntroduceF(clauses, numVarsC, numclauses, varsIntroduce, solsE, lastVars, weights, id, run.mode);
        if (tmp > 0) {
            double last = solsF->solutionCountFor(id);
            last = max(last, 0.0);
            if (!(run.mode & NO_EXP))  {
                solsF->setCount(id, (tmp / value + last));
                atomicMax(exponent, ilogb((tmp / value + last)));
                assert(*exponent != FP_ILOGB0);
            } else {
                solsF->setCount(id, (tmp + last));
            }
            //if (auto array = std::get_if<ArraySolution>(&solsF)) {
            // count solution (no-op except for ArraySolution)
            solsF->incSolutions();
            //}
        }
    } 
}

/**
 * Clones a solution bag to the GPU,
 * this clone uses the data of the <CudaMem> bag it was cloned from
 * and does not own it.
 */

template <template<typename> typename T>
std::unique_ptr<T<GpuOnly>, CudaMem> gpuClone(T<CudaMem>& owner) {
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
    TreeSolution<CpuMem>& to,
    const TreeSolution<CpuMem>& from,
    RunMeta meta
) {
    // copy to-tree and reserve additional space for the from-tree + connecting nodes
    auto to_owner = gpuOwner(to, from.currentTreeSize() + 2);
    auto from_owner = gpuOwner(from);

    auto to_gpu = gpuClone(to_owner);
    auto from_gpu = gpuClone(from_owner);

    int64_t threadsPerBlock = 512;
    int64_t threads = meta.maxId - meta.minId;
    int64_t blocksPerGrid = (threads + threadsPerBlock - 1) / threadsPerBlock;
    combineTree<<<blocksPerGrid, threadsPerBlock>>>(
        to_gpu.get(),
        from_gpu.get(),
        meta
    );
    gpuErrchk(cudaDeviceSynchronize());
    update(to_owner, to_gpu);
    return std::move(to_owner);
}

TreeSolution<CudaMem> array2treeWrapper(
    const ArraySolution<CpuMem>& array,
    int64_t *exponent,
    /// Number of variables in the resulting tree
    size_t tree_variables,
    RunMeta meta
) {
    
    auto max_tree_size = (array.maxId() - array.minId()) * 2 + tree_variables;
    TreeSolution<CpuMem> tmp(max_tree_size, array.minId(), array.maxId(), tree_variables);

    auto tree_owner = gpuOwner(tmp);
    auto array_owner = gpuOwner(array);

    auto tree_gpu = gpuClone(tree_owner);
    auto array_gpu = gpuClone(array_owner);

    int64_t threadsPerBlock = 512;
    int64_t threads = meta.maxId - meta.minId;
    int64_t blocksPerGrid = (threads + threadsPerBlock - 1) / threadsPerBlock;
    array2tree<<<blocksPerGrid, threadsPerBlock>>>(
        tree_gpu.get(),
        array_gpu.get(),
        exponent,
        meta
    );
    gpuErrchk(cudaDeviceSynchronize());
    update(tree_owner, tree_gpu);
    return std::move(tree_owner);
}

void solveJoinWrapper( 
    CudaSolutionVariant& solution_variant,
    const std::optional<SolutionVariant*> edge1,
    const std::optional<SolutionVariant*> edge2,
    GPUVars variables,
    GPUVars edgeVariables1,
    GPUVars edgeVariables2,
    double *weights,
    double value,
    int64_t *exponent,
    RunMeta meta
) {
    int64_t threadsPerBlock = 512;
    int64_t threads = meta.maxId - meta.minId;
    int64_t blocksPerGrid = (threads + threadsPerBlock - 1) / threadsPerBlock;

    assert(edge1.has_value() || edge2.has_value());

    std::visit([&](auto& solution) {
        auto solution_gpu = gpuClone(solution);

        auto single_edge_join = [&](SolutionVariant* edge) {
            assert(edge != nullptr);
            std::visit([&](auto &sol) {
                auto edge_owner = gpuOwner(sol);
                auto edge_gpu = gpuClone(edge_owner);

                solveJoin<<<blocksPerGrid, threadsPerBlock>>>(
                    solution_gpu.get(),
                    edge_gpu.get(),
                    (decltype(edge_owner)*)nullptr,
                    variables,
                    edgeVariables1,
                    edgeVariables2,
                    weights,
                    value, 
                    exponent,
                    meta
                );
            }, *edge);
            gpuErrchk(cudaDeviceSynchronize());
        };

        auto double_edge_join = [&](SolutionVariant* e1, SolutionVariant* e2) {
            assert(e1 != nullptr);
            assert(e2 != nullptr);
            std::visit([&](auto &sol1) {
                auto e1_owner = gpuOwner(sol1);
                auto e1_gpu = gpuClone(e1_owner);

                std::visit([&](auto &sol2) {
                    auto e2_owner = gpuOwner(sol2);
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
                        meta
                    );
                }, *e2);
            }, *e1);
            gpuErrchk(cudaDeviceSynchronize());
        };

        if (edge1.has_value() && !edge2.has_value()) {
            single_edge_join(edge1.value());
        } else if (!edge1.has_value() && edge2.has_value()) {
            single_edge_join(edge2.value());
        } else {
            assert(edge1.has_value() && edge2.has_value());
            double_edge_join(edge1.value(), edge2.value());
        }
        update(solution, solution_gpu);
    }, solution_variant);
}

void introduceForgetWrapper(
    std::variant<TreeSolution<CudaMem>, ArraySolution<CudaMem>>& solution_owner,
    GPUVars varsForget,
    const std::optional<SolutionVariant*> edge,
    GPUVars lastVars,
    GPUVars varsIntroduce,
    // FIXME: Move this static information to GPU once.
    long *clauses,
    long *numVarsC,
    long numclauses,
    double *weights,
    int64_t *exponent,
    double previous_value,
    RunMeta meta
) {
    int64_t threadsPerBlock = 512;
    int64_t threads = meta.maxId - meta.minId;
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
                numVarsC,
                numclauses,
                weights,
                exponent, 
                previous_value,
                meta
            );
        } else {
            assert(edge.value() != nullptr);
            std::visit([&](auto& solsE) {
                auto edge_owner = gpuOwner(solsE);
                auto edge_gpu = gpuClone(edge_owner);
 
                solveIntroduceForget<<<blocksPerGrid, threadsPerBlock>>>(
                    solution_gpu.get(),
                    varsForget,
                    edge_gpu.get(),
                    lastVars,
                    combinations,
                    varsIntroduce,
                    clauses,
                    numVarsC,
                    numclauses,
                    weights,
                    exponent, 
                    previous_value,
                    meta
                );
            }, *edge.value());
        }
        gpuErrchk(cudaDeviceSynchronize());
        update(solsF, solution_gpu);
    }, solution_owner);
}

/**
 * Returns a solution bag that lives on the CPU
 * but owns GPU data copied from the input bag.
 */
template <template<typename> typename T>
T<CudaMem> gpuOwner(const T<CpuMem>& orig, size_t reserve) {
    // copy parameters
    T<CudaMem> gpu(orig, nullptr);

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
            auto start = thrust::device_ptr<decltype(orig.initializer())>(gpu.data());
            thrust::fill(
                start + orig.dataStructureSize(),
                start + gpu.dataStructureSize(),
                orig.initializer()
            );
        }
    } else {
        auto start = thrust::device_ptr<decltype(orig.initializer())>(gpu.data());
        thrust::fill(start, start + gpu.dataStructureSize(), orig.initializer());
    }
    return std::move(gpu);
}

CudaSolutionVariant gpuOwner(const SolutionVariant& orig, size_t reserve) {
    return std::visit([](auto& sol) -> CudaSolutionVariant {
        return std::variant<TreeSolution<CudaMem>, ArraySolution<CudaMem>>(std::move(gpuOwner(sol)));
    }, orig);
}

}
