#ifndef KERNEL_H
#define KERNEL_H

#include <gpusat_types.h>
#include <thrust/fill.h>
#include <thrust/device_ptr.h>

namespace gpusat {
    TreeSolution<CudaMem> combineTreeWrapper(
        TreeSolution<CudaMem>& to_owner,
        const TreeSolution<CudaMem>& from_owner
    );

    void solveJoinWrapper(
        CudaSolutionVariant& solution,
        const std::optional<CudaSolutionVariant>& edge1,
        const std::optional<CudaSolutionVariant>& edge2,
        GPUVars variables,
        GPUVars edgeVariables1,
        GPUVars edgeVariables2,
        double *weights,
        double value,
        int32_t *exponent,
        const SolveConfig cfg
    );

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
        double value,
        const SolveConfig cfg
    );

    // Ensure to instantiate explicitly in kernel.cu for needed types.
    template <template<typename> typename T>
    T<CpuMem> cpuCopy(const T<CudaMem>& gpu, size_t reserve = 0);

    SolutionVariant cpuCopy(const CudaSolutionVariant& gpu, size_t reserve = 0);

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
