#include <gpusat.h>
#include "decomposer.h"
#include "gpusatparser.h"
#include "gpusatpreprocessor.h"
#include "solver.h"
#include "kernel.h"
#include <cuda_runtime.h>

namespace gpusat {


    // The pinned memory suballocator to use for solution bags.
    PinnedSuballocator cuda_pinned_alloc_pool(false);

    treedecType GPUSAT::decompose(
        const satformulaType& formula,
        htd::ITreeDecompositionFitnessFunction& fitness,
        size_t iterations,
        bool dumpDecomp
    ) {
        return Decomposer::computeDecomposition(formula, &fitness, iterations, dumpDecomp);
    }

    std::pair<PreprocessingResult, double> GPUSAT::preprocessFormula(satformulaType& formula, std::vector<int64_t>& removed_facts) {

        double weight_correction = GPUSAT::default_variable_weight;
        Preprocessor::preprocessFacts(formula, weight_correction);
        removed_facts.reserve(formula.facts.size());
        std::copy(formula.facts.begin(), formula.facts.end(), back_inserter(removed_facts));
        std::sort(removed_facts.begin(), removed_facts.end(), compVars);
        // We extract the facts before relabeling
        Preprocessor::relabelFormula(formula);
        formula.facts.clear();
        if (formula.unsat) {
            return std::pair(PreprocessingResult::UNSATISFIABLE, GPUSAT::default_variable_weight);
        }
        std::cout << "facts preprocessed." << std::endl;
        return std::pair(PreprocessingResult::SUCCESS, weight_correction);
    }


    PreprocessingResult GPUSAT::preprocessDecomp(
        treedecType& decomposition,
        size_t combine_width
    ) {
        Preprocessor::preprocessDecomp(decomposition.root, combine_width);
        return PreprocessingResult::SUCCESS;
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
            cfg.trace,
            cfg.gpu_cache
        );
        auto& root = decomposition.root;

#ifndef NDEBUG
        Preprocessor::checkNoFactInDecomp(root, formula.facts);
#endif

        auto dummy_parent = BagType();
        dummy_parent.variables.assign(
                root.variables.begin(),
                root.variables.begin()
                // FIXME: why 12??
                    + std::min(root.variables.size(), 12ul)
        );
        solver.solveProblem(formula, root, dummy_parent, INTRODUCEFORGET);
        if (root.cached_solution.has_value()) {
            root.solution.push_back(cpuCopy(root.cached_solution.value()));
        }

        boost::multiprecision::cpp_bin_float_100 sols = 0.0;
        if (solver.isSat > 0) {
            sols += Solver::bagSum(root);
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
