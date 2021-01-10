#define __CL_ENABLE_EXCEPTIONS

// do not extend function signatures for CUDA.
#define GPU_HOST_ATTR

#include <iostream>
#include <sstream>
#include <fstream>
#include <regex>
#include <math.h>
#include <chrono>
#include <sys/stat.h>
#include <numeric>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include "CLI11.hpp"


#include <gpusat.h>
#include <gpusatparser.h>
#include "FitnessFunctions/NumJoinFitnessFunction.h"
#include "FitnessFunctions/JoinSizeFitnessFunction.h"
#include "FitnessFunctions/WidthCutSetFitnessFunction.h"
#include "FitnessFunctions/CutSetWidthFitnessFunction.h"

using namespace gpusat;

/**
 * @return the time in millisecons since the epoch
 */
inline long long int getTime() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

int main(int argc, char *argv[]) {
    long long int time_total = getTime();
    std::string inputLine;
    std::string formulaDir;
    std::string fitness;
    std::string type;
    std::string decompDir;
    size_t combineWidth = 0;
    time_t seed = time(0);
    bool weighted, noExp, trace, no_cache, unpinned;
    dataStructure solutionType = dataStructure::TREE;
    CLI::App app{};
    std::size_t numDecomps = 30;
    size_t maxBag = 0;

    //cmd options
    app.add_option("-s,--seed", seed, "path to the file containing the sat formula")->set_default_str("");
    app.add_option("-f,--formula", formulaDir, "path to the file containing the sat formula")->set_default_str("");
    app.add_option("-d,--decomposition", decompDir, "path to the file containing the tree decomposition")->set_default_str("");
    app.add_option("-n,--numDecomps", numDecomps, "")->set_default_str("30");
    app.add_set("--fitnessFunction", fitness, {"numJoin", "joinSize", "width_cutSet", "cutSet_width"},
                "fitness functions:\n"
                "\t\t\tnumJoin: minimize the number of joins\n"
                "\t\t\tjoinSize: minimize the numer of variables in a join node\n"
                "\t\t\twidth_cutSet: minimize the width and then the cut set size\n"
                "\t\t\tcutSet_width: minimize the cut set size and then the width")->set_default_str("width_cutSet");
    app.add_flag("--weighted", weighted, "use weighted model count");
    app.add_flag("--unpinned", unpinned, "do not use pinned memory for solutions");
    app.add_flag("--noExp", noExp, "don't use extended exponents");
    app.add_flag("--trace", trace, "output a solver trace");
    app.add_flag("--no-cache", no_cache, "cache solution bags on GPU when possible");
    app.add_set("--dataStructure", type, {"array", "tree", "combined"}, "data structure for storing the solution")->set_default_str("combined");
    app.add_option("-m,--maxBagSize", maxBag, "max size of a bag on the gpu")->set_default_str("0");
    app.add_option("-w,--combineWidth", combineWidth, "maximum width to combine bags of the decomposition")->set_default_str("0");
    CLI11_PARSE(app, argc, argv)

    GPUSAT gsat(!unpinned);

    if (combineWidth == 0) {
        combineWidth = gsat.recommended_bag_width();
    }

    /*
     * Build configuration.
     */
    GpusatConfig cfg;

    cfg.solve_cfg.no_exponent = noExp;
    cfg.solve_cfg.weighted = weighted;
    cfg.trace = trace;
    cfg.gpu_cache = !no_cache;
    cfg.max_bag_size = maxBag;

    std::srand(seed);

    satformulaType formula;
    treedecType decomposition;
    time_t time_decomposing;
    {

        CNFParser cnf_parser(cfg.solve_cfg.weighted);
        if (formulaDir != "") {
            std::ifstream fileIn(formulaDir);
            formula = cnf_parser.parseSatFormula(fileIn);
        } else {
            formula = cnf_parser.parseSatFormula(std::cin);
        }

        std::cout << "formula parsed: " << formula.facts.size() << " " << formula.clause_offsets.size() << " " << formula.clause_bag.size() << std::endl;

        time_decomposing = getTime();
        if (decompDir != "") {
            // FIXME: implement through htd_io
        } else {
            htd::ITreeDecompositionFitnessFunction *fit;
            if (fitness == "width") {
                fit = new NumJoinFitnessFunction();
            } else if (fitness == "cutSet") {
                fit = new JoinSizeFitnessFunction();
            } else if (fitness == "cutSet_width") {
                fit = new CutSetWidthFitnessFunction();
            } else {
                fit = new WidthCutSetFitnessFunction();
            }
            std::cout << "num decomps: " << numDecomps << std::endl;
            decomposition = GPUSAT::decompose(formula, *fit, numDecomps);
        }
        time_decomposing = getTime() - time_decomposing;
        std::cout << "decomp time: " << time_decomposing / 1000.0 << std::endl;
    }

    if (type == "array") {
        cfg.solution_type = dataStructure::ARRAY;
    } else if (type == "tree") {
        cfg.solution_type = dataStructure::TREE;
    } else if (type == "combined" || type == "") {
        if (decomposition.width < 25) {
            cfg.solution_type = dataStructure::ARRAY;
        } else {
            cfg.solution_type = dataStructure::TREE;
        }
    } else {
        std::cerr << "unknown data structure: " << type << std::endl;
        return 1;
    }

    std::cerr << "before pp: " << decomposition.root.hash() << std::endl;

    auto time_pp = getTime();
    auto pp_result = GPUSAT::preprocess(formula, decomposition, combineWidth);
    std::cout << "pp time: " << (getTime() - time_pp) / 1000.0 << std::endl;
    std::cerr << "after pp: " << decomposition.root.hash() << std::endl;

    std::cout << "formula preprocessed: " << formula.facts.size() << " " << formula.clause_offsets.size() << " " << formula.clause_bag.size() << std::endl;

    auto weight_correction = pp_result.second;
    if (pp_result.first != PreprocessingResult::SUCCESS) {
        time_total = getTime() - time_total;
        std::cout << "\n{\n";
        std::cout << "    \"Num Join\": " << 0;
        std::cout << "\n    ,\"Num Introduce Forget\": " << 0;
        std::cout << "\n    ,\"max Table Size\": " << 0;
        std::cout << "\n    ,\"Model Count\": " << 0;
        std::cout << "\n    ,\"Time\":{";
        std::cout << "\n        \"Decomposing\": " << ((float) time_decomposing) / 1000;
        std::cout << "\n        ,\"Solving\": " << 0;
        std::cout << "\n        ,\"Total\": " << ((float) time_total) / 1000;
        std::cout << "\n    }";
        std::cout << "\n}\n";
        return 20;
    }

    //cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1 << 26);

    time_t time_solving = getTime();

    auto model_count = gsat.solve(formula, decomposition, cfg, weight_correction);

    time_t time_finish = getTime();
    time_solving = time_finish - time_solving;
    time_total = time_finish - time_total;

    /*
    std::cout << "\n{\n";
    std::cout << "    \"Num Join\": " << sol->numJoin;
    std::cout << "\n    ,\"Num Introduce Forget\": " << sol->numIntroduceForget;
    std::cout << "\n    ,\"max Table Size\": " << sol->maxTableSize;
    */
    std::cout << std::setprecision(20) << "\n    ,\"Model Count\": " << model_count;

    std::cout.precision(6);
    std::cout << "\n    ,\"Time\":{";
    std::cout << "\n        \"Decomposing\": " << time_decomposing / 1000.0;
    std::cout << "\n        ,\"Solving\": " << time_solving / 1000.0;
    std::cout << "\n        ,\"Total\": " << time_total / 1000.0;
    std::cout << "\n    }";
    std::cout << "\n}\n";

    if (model_count > 0.0) {
        return 10;
    } else {
        return 20;
    }
}

