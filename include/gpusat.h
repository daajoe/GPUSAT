#ifndef GPUSAT_H
#define GPUSAT_H

#define GPU_HOST_ATTR
#include <gpusat_types.h>

#include <htd/ITreeDecompositionFitnessFunction.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

namespace gpusat {
    struct GpusatConfig {
        /// The solution data structure to use
        dataStructure solution_type;
        /// Configuration concerning the solving process
        SolveConfig solve_cfg;
        /// Output a trace of the solving process
        bool trace;
        /// Cache solution bags in GPU mem
        bool gpu_cache;
        /// Maximum size of a solution container on the GPU.
        /// Pass 0 to not restrict.
        size_t max_bag_size;
    };

    /**
     * Possible outcomes of a preprocessing step.
     */
    enum PreprocessingResult {
        SUCCESS,
        UNSATISFIABLE
    };

    class GPUSAT {
        public:
            /**
             * Setup the GPU context.
             */
            GPUSAT(bool use_pinned_memory) {
                cuda_pinned_alloc_pool.deinit();
                cuda_pinned_alloc_pool = std::move(PinnedSuballocator(use_pinned_memory));

                cudaDeviceProp deviceProp;
                cudaGetDeviceProperties(&deviceProp, 0);
                memory_size = deviceProp.totalGlobalMem;
                // FIXME: is 1 / 4 in opencl, calculations do not properly account for
                // full mem beeing available
                max_memory_buffer = deviceProp.totalGlobalMem / 4;
                recommended_bag_width_ = std::floor(std::log2(deviceProp.maxThreadsPerBlock * deviceProp.multiProcessorCount));
            }

            ~GPUSAT() {
                // free the pinned memory pool.
                cuda_pinned_alloc_pool.deinit();
            }

            /**
             * The recommended bag size for this graphics card.
             */
            size_t recommended_bag_width() {
	        return recommended_bag_width_;
            }

            /**
             * Compute a tree decomposition of `formula` for the use in GPUSAT.
             * `formula` must be *fact-free*!
             *
             * @param formula: The formula to decompose.
             * @param fitness: The fitness function to use.
             * @param iterations: Number of iterations to improve upon the decomposition.
             * @param dumpDecomp: Dump the tree decomposition in td format and exit.
             */
            static treedecType decompose(const satformulaType& formula, htd::ITreeDecompositionFitnessFunction& fitness, size_t iterations, bool dumpDecomp);

            /**
             * Apply preprocessing steps to a tree decomposition to allow for more
             * efficient solving.
             *
             * @param decomposition: A decomposition of a fact-free formula.
             * @param combine_width: Maximal width of bags that can be combined.
             *
             * @returns: Pair of a result and weight correction. The weight correction
             * is needed to calculate the correct final results for weighted counting.
             */
            static PreprocessingResult preprocessDecomp(treedecType& decomposition, size_t combine_width);

            /**
             * Apply fact propagation to a formula.
             * The resulting formula is fact-free.
             *
             * @param formula: The sat formula.
             * @param removed_facts: A vector to store the removed facts in.
             *
             * @returns: Pair of a result and weight correction. The weight correction
             * is needed to calculate the correct final results for weighted counting.
             */
            static std::pair<PreprocessingResult, double> preprocessFormula(satformulaType& formula, std::vector<int64_t>& removed_facts);

            /**
             * Count the models of `formula` with a given tree decomposition
             * and configuration.
             *
             * @param formula: The formula to count models for.
             * @param treedecType: A tree decomposition for `formula`.
             * @param cfg: Configuration of the solver's behaviour.
             */
            boost::multiprecision::cpp_bin_float_100 solve(
                    const satformulaType& formula,
                    treedecType& decomposition,
                    const GpusatConfig cfg,
                    double weight_correction = GPUSAT::default_variable_weight
            ) const;

            /// Default weight for a variable.
            static constexpr const double default_variable_weight = 1.0;
        protected:
            /// Total size of GPU memory.
            size_t memory_size;
            /// Maximum size of a buffer on the GPU.
            size_t max_memory_buffer;
            /// Recommended bag width
            size_t recommended_bag_width_;
    };

}

#endif // GPUSAT_H
