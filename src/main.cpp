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
#include <cuda_runtime.h>

#include "decomposer.h"
#include "solver.h"
#include "CLI11.hpp"
#include "types.h"
#include "gpusatparser.h"
#include "gpusatutils.h"
#include "gpusatpreprocessor.h"
#include "alloc.h"

#include "FitnessFunctions/NumJoinFitnessFunction.h"
#include "FitnessFunctions/JoinSizeFitnessFunction.h"
#include "FitnessFunctions/WidthCutSetFitnessFunction.h"
#include "FitnessFunctions/CutSetWidthFitnessFunction.h"

using namespace gpusat;

template<class... Ts> struct free_visitor : Ts... { using Ts::operator()...; };
template<class... Ts> struct sum_visitor : Ts... { using Ts::operator()...; };
// explicit deduction guide (not needed as of C++20)
template<class... Ts> free_visitor(Ts...) -> free_visitor<Ts...>;
template<class... Ts> sum_visitor(Ts...) -> sum_visitor<Ts...>;

template<class T>
boost::multiprecision::cpp_bin_float_100 solutionSum(T& sol) {
    boost::multiprecision::cpp_bin_float_100 sols = 0.0;
    for (size_t i = sol.minId(); i < sol.maxId(); i++) {
        sols = sols + std::max(sol.solutionCountFor(i), 0.0);
    }
    return sols;
}

boost::multiprecision::cpp_bin_float_100 bag_sum(BagType& bag) {
    boost::multiprecision::cpp_bin_float_100 sols = 0.0;
    for (auto& solution : bag.solution) {
        sols = sols + std::visit([](auto& sol) { return solutionSum(sol); }, solution);
    }
    return sols;
}

namespace gpusat {
    // The pinned memory suballocator to use for solution bags.
    PinnedSuballocator cuda_pinned_alloc_pool;
}

int main(int argc, char *argv[]) {
    long long int time_total = getTime();
    std::string inputLine;
    std::string formulaDir;
    std::string fitness;
    std::string type;
    std::string decompDir;
    size_t combineWidth = 0;
    time_t seed = 2;
    bool cpu, weighted, noExp, nvidia, amd;
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
    app.add_flag("--CPU", cpu, "run the solver on a cpu");
    app.add_flag("--NVIDIA", nvidia, "run the solver on an NVIDIA device");
    app.add_flag("--AMD", amd, "run the solver on an AMD device");
    app.add_flag("--weighted", weighted, "use weighted model count");
    app.add_flag("--noExp", noExp, "don't use extended exponents");
    app.add_set("--dataStructure", type, {"array", "tree", "combined"}, "data structure for storing the solution")->set_default_str("combined");
    app.add_option("-m,--maxBagSize", maxBag, "max size of a bag on the gpu")->set_default_str("0");
    app.add_option("-w,--combineWidth", maxBag, "maximum width to combine bags of the decomposition")->set_default_str("0");
    CLI11_PARSE(app, argc, argv)

    std::srand(seed);


    satformulaType satFormula;
    treedecType treeDecomp;
    TDParser tdParser;
    long long int time_decomposing;
    {
        std::stringbuf treeD, sat;
        if (formulaDir != "") {
            std::ifstream fileIn(formulaDir);
            while (getline(fileIn, inputLine)) {
                sat.sputn(inputLine.c_str(), inputLine.size());
                sat.sputn("\n", 1);
            }
        } else {
            while (getline(std::cin, inputLine)) {
                sat.sputn(inputLine.c_str(), inputLine.size());
                sat.sputn("\n", 1);
            }
        }

        time_decomposing = getTime();
        std::string treeDString;
        if (decompDir != "") {
            std::ifstream fileIn(decompDir);
            while (getline(fileIn, inputLine)) {
                treeD.sputn(inputLine.c_str(), inputLine.size());
                treeD.sputn("\n", 1);
            }
            treeDString = treeD.str();
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
            treeDString = Decomposer::computeDecomposition(sat.str(), fit, numDecomps);
        }
        time_decomposing = getTime() - time_decomposing;

        std::string satString = sat.str();

        if (satString.size() < 8) {
            std::cerr << "Error: SAT formula\n";
            cuda_pinned_alloc_pool.deinit();
            exit(EXIT_FAILURE);
        }
        if (treeDString.size() < 8) {
            std::cerr << "Error: tree decomposition\n";
            cuda_pinned_alloc_pool.deinit();
            exit(EXIT_FAILURE);
        }
        satFormula = CNFParser(weighted).parseSatFormula(sat.str());
        treeDecomp = tdParser.parseTreeDecomp(treeDString, satFormula);
    }
    std::cout.flush();

    Preprocessor::preprocessFacts(treeDecomp, satFormula, tdParser.defaultWeight);
    if (satFormula.unsat) {
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
        cuda_pinned_alloc_pool.deinit();
        exit(20);
    }

    SolveMode solve_mode = SolveMode::DEFAULT;
    if (noExp) {
        solve_mode = solve_mode | SolveMode::NO_EXP;
    }
    if (type == "array") {
        solve_mode = solve_mode | SolveMode::ARRAY_TYPE;
        solutionType = dataStructure::ARRAY;
    } else if (type == "tree") {
        solutionType = dataStructure::TREE;
    } else if (type == "combined") {
        if (treeDecomp.width < 30) {
            solve_mode = solve_mode | SolveMode::ARRAY_TYPE;
            solutionType = dataStructure::ARRAY;
        } else {
            solutionType = dataStructure::TREE;
        }
    }

    //std::cerr << "solve mode: " << solve_mode << std::endl;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int64_t memorySize = deviceProp.totalGlobalMem;
    // FIXME: is 1 / 4 in opencl, calculations do not properly account for
    // full mem beeing available
    int64_t maxMemoryBuffer = deviceProp.totalGlobalMem / 4;

    if (combineWidth == 0) {
	std::cerr << "maximum workgroup size: " << deviceProp.maxThreadsPerBlock << " " << deviceProp.multiProcessorCount << std::endl;
	combineWidth = (long) std::floor(std::log2(deviceProp.maxThreadsPerBlock * deviceProp.multiProcessorCount));
    }

    //cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1 << 26);
    //buildKernel(context, devices, queue, program, memorySize, maxMemoryBuffer, nvidia, amd, cpu, combineWidth);

    // combine small bags
    std::cerr << "before pp: " << treeDecomp.bags[0].hash() << std::endl;
    Preprocessor::preprocessDecomp(treeDecomp.bags[0], combineWidth);
    std::cerr << "after pp: " << treeDecomp.bags[0].hash() << std::endl;

    std::cout.flush();

    Solver *sol;
    auto next = BagType();
    sol = new Solver(memorySize, maxMemoryBuffer, solutionType, maxBag, solve_mode);

    next.variables.assign(
            treeDecomp.bags[0].variables.begin(),
            treeDecomp.bags[0].variables.begin()
                + std::min((int64_t) treeDecomp.bags[0].variables.size(), (int64_t)12));
    long long int time_solving = getTime();
    (*sol).solveProblem(satFormula, treeDecomp.bags[0], next, INTRODUCEFORGET);
    time_solving = getTime() - time_solving;

    std::cout << "\n{\n";
    std::cout << "    \"Num Join\": " << sol->numJoin;
    std::cout << "\n    ,\"Num Introduce Forget\": " << sol->numIntroduceForget;
    std::cout << "\n    ,\"max Table Size\": " << sol->maxTableSize;

    //sum up last node solutions
    long long int time_model = getTime();
    boost::multiprecision::cpp_bin_float_100 sols = 0.0;
    if ((*sol).isSat > 0) {
        sols += bag_sum(treeDecomp.bags[0]);
        for (auto& solution : treeDecomp.bags[0].solution) {
            std::visit([](auto& sol) {sol.freeData(); }, solution);
        }

        if (!noExp) {
            boost::multiprecision::cpp_bin_float_100 base = 2;
            boost::multiprecision::cpp_bin_float_100 exponent = treeDecomp.bags[0].correction;
            sols = sols * pow(base, exponent);
        }

        if (weighted) {
            sols = sols * tdParser.defaultWeight;
        }

        std::cout << std::setprecision(20) << "\n    ,\"Model Count\": " << sols;

    } else {
        std::cout << "\n    ,\"Model Count\": " << 0;
    }
    time_model = getTime() - time_model;
    time_total = getTime() - time_total;
    std::cout.precision(6);
    std::cout << "\n    ,\"Time\":{";
    std::cout << "\n        \"Decomposing\": " << ((float) time_decomposing) / 1000;
    std::cout << "\n        ,\"Solving\": " << ((float) time_solving) / 1000;
    std::cout << "\n        ,\"Total\": " << ((float) time_total) / 1000;
    std::cout << "\n    }";
    std::cout << "\n}\n";
    std::cout.flush();
    cuda_pinned_alloc_pool.deinit();
    if (sols > 0) {
        exit(10);
    } else {
        exit(20);
    }
}

