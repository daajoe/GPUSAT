#ifndef KERNEL_H
#define KERNEL_H

#include "types.h"
#include <thrust/fill.h>
#include <thrust/device_ptr.h>


namespace gpusat {
    TreeSolution<CudaMem> combineTreeWrapper(
        TreeSolution<CpuMem>& to,
        const TreeSolution<CpuMem>& from,
        RunMeta meta
    );

    TreeSolution<CudaMem> array2treeWrapper(
        const ArraySolution<CpuMem>& array,
        int64_t *exponent,
        /// Number of variables in the resulting tree
        size_t tree_variables,
        /// Only the `mode` is currently used.
        RunMeta meta
    );

    void solveJoinWrapper(
        CudaSolutionVariant& solution,
        const std::optional<SolutionVariant*> edge1,
        const std::optional<SolutionVariant*> edge2,
        GPUVars variables,
        GPUVars edgeVariables1,
        GPUVars edgeVariables2,
        double *weights,
        double value,
        int64_t *exponent,
        RunMeta meta
    );

    void introduceForgetWrapper(
        CudaSolutionVariant& solution_owner,
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
        double value,
        RunMeta meta
    );

    /**
     * Returns a solution bag that is constructed
     * (copied) from a bag tat owns GPU data.
     */
    template <template<typename> typename T>
    T<CpuMem> cpuCopy(const T<CudaMem>& gpu, size_t reserve = 0) {
        // copy parameters
        T<CpuMem> cpu(gpu, nullptr);

        cpu.setDataStructureSize(cpu.dataStructureSize() + reserve);

        // allocate CPU memory
        cpu.allocate();

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
        return std::move(cpu);
    }

    /**
     * Updates a solution bag with information
     * that may have changed on its GPU clone.
     */
    template <template<typename> typename T>
    void update(T<CudaMem>& owner, const std::unique_ptr<T<GpuOnly>, CudaMem>& gpu) {
        static_assert(sizeof(T<CudaMem>) == sizeof(T<GpuOnly>));

        gpuErrchk(cudaMemcpy(&owner, gpu.get(), sizeof(T<GpuOnly>), cudaMemcpyDeviceToHost));
    }

    /**
     * Returns a solution bag that lives on the CPU
     * but owns GPU data copied from the input bag.
     */
    template <template<typename> typename T>
    T<CudaMem> gpuOwner(const T<CpuMem>& orig, size_t reserve = 0);

    CudaSolutionVariant gpuOwner(const SolutionVariant& orig, size_t reserve = 0);

}
#endif // KERNEL_H
