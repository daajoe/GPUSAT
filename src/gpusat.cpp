#include <gpusat.h>
#include "decomposer.h"
#include "gpusatparser.h"
#include "gpusatpreprocessor.h"
#include "solver.h"
#include <cuda_runtime.h>

namespace gpusat {

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

    static boost::multiprecision::cpp_bin_float_100 bag_sum(BagType& bag) {
        boost::multiprecision::cpp_bin_float_100 sols = 0.0;
        for (auto& solution : bag.solution) {
            sols = sols + std::visit([](auto& sol) { return solutionSum(sol); }, solution);
        }
        return sols;
    }


    // The pinned memory suballocator to use for solution bags.
    PinnedSuballocator cuda_pinned_alloc_pool;

    treedecType GPUSAT::decompose(
        const satformulaType& formula,
        htd::ITreeDecompositionFitnessFunction& fitness,
        size_t iterations
    ) {
        return Decomposer::computeDecomposition(formula, &fitness, iterations);
    }

    std::pair<PreprocessingResult, double> GPUSAT::preprocess(
        satformulaType& formula,
        treedecType& decomposition,
        size_t combine_width
    ) {
        double weight_correction = GPUSAT::default_variable_weight;
        Preprocessor::preprocessFacts(decomposition, formula, weight_correction);
        if (formula.unsat) {
            return std::pair(PreprocessingResult::UNSATISFIABLE, GPUSAT::default_variable_weight);
        }
        std::cout << "facts preprocessed." << std::endl;

        Preprocessor::preprocessDecomp(decomposition.root, combine_width);
        return std::pair(PreprocessingResult::SUCCESS, weight_correction);
    }

    boost::multiprecision::cpp_bin_float_100 GPUSAT::solve(
            const satformulaType& formula,
            treedecType& decomposition,
            const GpusatConfig cfg,
            double weight_correction
    ) const {
        Solver solver(
            memory_size,
            max_memory_buffer,
            cfg.solution_type,
            cfg.max_bag_size,
            cfg.solve_cfg,
            cfg.trace
        );

        auto& root = decomposition.root;
        auto dummy_parent = BagType();
        dummy_parent.variables.assign(
                root.variables.begin(),
                root.variables.begin()
                // FIXME: why 12??
                    + std::min(root.variables.size(), 12ul)
        );
        solver.solveProblem(formula, root, dummy_parent, INTRODUCEFORGET);

        boost::multiprecision::cpp_bin_float_100 sols = 0.0;
        if (solver.isSat > 0) {
            sols += bag_sum(root);
            // free solutions
            for (auto& solution : root.solution) {
                std::visit([](auto& sol) {sol.freeData(); }, solution);
            }
            root.cached_solution = std::nullopt;

            if (!cfg.solve_cfg.no_exponent) {
                boost::multiprecision::cpp_bin_float_100 base = 2;
                boost::multiprecision::cpp_bin_float_100 exponent = root.correction;
                sols = sols * pow(base, exponent);
            }

            if (cfg.solve_cfg.weighted) {
                sols = sols * weight_correction;
            }
        }
        return sols;
    }
}
