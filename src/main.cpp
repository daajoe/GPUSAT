#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <sstream>
#include <fstream>
#include <regex>
#include <math.h>
#include <chrono>
#include <types.h>
#include <gpusatparser.h>
#include <gpusatutils.h>
#include <sys/stat.h>
#include <numeric>
#include <decomposer.h>
#include <solver.h>
#include <CLI11.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <gpusatpreprocessor.h>
#include <FitnessFunctions/NumJoinFitnessFunction.h>
#include <FitnessFunctions/JoinSizeFitnessFunction.h>
#include <FitnessFunctions/WidthCutSetFitnessFunction.h>
#include <FitnessFunctions/CutSetWidthFitnessFunction.h>

std::string kernelStr =

#include <kernel.h>

using namespace gpusat;

void buildKernel(cl::Context &context, std::vector<cl::Device> &devices, cl::CommandQueue &queue, cl::Program &program, cl_long &memorySize, cl_long &maxMemoryBuffer, bool nvidia, bool amd, bool cpu, int &combineWidth) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    std::vector<cl::Platform>::iterator iter;
    for (iter = platforms.begin(); iter != platforms.end(); ++iter) {
        if (nvidia && amd) {
            if (strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "NVIDIA Corporation") && strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "Advanced Micro Devices, Inc.")) {
                continue;
            }
        } else if (nvidia) {
            if (strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "NVIDIA Corporation")) {
                continue;
            }
        } else if (amd) {
            if (strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "Advanced Micro Devices, Inc.")) {
                continue;
            }
        }
        cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties) (*iter)(), 0};
        if (cpu) {
            context = cl::Context(CL_DEVICE_TYPE_CPU, cps);
        } else {
            context = cl::Context(CL_DEVICE_TYPE_GPU, cps);
        }
        cl_int err;
        devices = context.getInfo<CL_CONTEXT_DEVICES>(&err);
        if (err == CL_SUCCESS) {
            queue = cl::CommandQueue(context, devices[0]);
            memorySize = devices[0].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
            maxMemoryBuffer = devices[0].getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
            if (combineWidth < 0) {
                combineWidth = (int) std::floor(std::log2(devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() * devices[0].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()));
            }
            break;
        }
    }
    if (iter == platforms.end()) {
        std::cout << "\nERROR: no GPU found!";
        exit(1);
    }

    cl::Program::Sources sources(1, std::make_pair(kernelStr.c_str(), kernelStr.length()));
    program = cl::Program(context, sources);
    program.build(devices);
}

int main(int argc, char *argv[]) {
    long long int time_total = getTime();
    std::string inputLine;
    std::string formulaDir;
    std::string fitness;
    std::string type;
    std::string decompDir;
    int combineWidth = -1;
    time_t seed = time(0);
    bool cpu, weighted, noExp, nvidia, amd;
    dataStructure solutionType = dataStructure::TREE;
    CLI::App app{};
    std::size_t numDecomps = 30;
    cl_long maxBag = -1;

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
    app.add_option("-m,--maxBagSize", maxBag, "max size of a bag on the gpu")->set_default_str("-1");
    app.add_option("-w,--combineWidth", maxBag, "maximum width to combine bags of the decomposition")->set_default_str("-1");
    CLI11_PARSE(app, argc, argv)

    srand(seed);

    if (noExp) {
        kernelStr = "#define NO_EXP\n" + kernelStr;
    }

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
            exit(EXIT_FAILURE);
        }
        if (treeDString.size() < 8) {
            std::cerr << "Error: tree decomposition\n";
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
        exit(20);
    }

    if (type == "array") {
        kernelStr = "#define ARRAY_TYPE\n" + kernelStr;
        solutionType = dataStructure::ARRAY;
    } else if (type == "tree") {
        solutionType = dataStructure::TREE;
    } else if (type == "combined") {
        if (treeDecomp.width < 30) {
            kernelStr = "#define ARRAY_TYPE\n" + kernelStr;
            solutionType = dataStructure::ARRAY;
        } else {
            solutionType = dataStructure::TREE;
        }
    }

    cl::Context context;
    std::vector<cl::Device> devices;
    cl::CommandQueue queue;
    cl::Program program;
    cl_long memorySize = 0;
    cl_long maxMemoryBuffer = 0;

    try {
        buildKernel(context, devices, queue, program, memorySize, maxMemoryBuffer, nvidia, amd, cpu, combineWidth);
        // combine small bags
        Preprocessor::preprocessDecomp(&treeDecomp.bags[0], combineWidth);

        std::cout.flush();

        Solver *sol;
        bagType next;
        sol = new Solver(context, queue, program, memorySize, maxMemoryBuffer, solutionType, maxBag);
        next.variables.assign(treeDecomp.bags[0].variables.begin(), treeDecomp.bags[0].variables.begin() + std::min((cl_long) treeDecomp.bags[0].variables.size(), (cl_long) 12));
        long long int time_solving = getTime();
        (*sol).solveProblem(treeDecomp, satFormula, treeDecomp.bags[0], next, INTRODUCEFORGET);
        time_solving = getTime() - time_solving;

        std::cout << "\n{\n";
        std::cout << "    \"Num Join\": " << sol->numJoin;
        std::cout << "\n    ,\"Num Introduce Forget\": " << sol->numIntroduceForget;
        std::cout << "\n    ,\"max Table Size\": " << sol->maxTableSize;

        //sum up last node solutions
        long long int time_model = getTime();
        boost::multiprecision::cpp_bin_float_100 sols = 0.0;
        if ((*sol).isSat > 0) {
            for (cl_long a = 0; a < treeDecomp.bags[0].bags; a++) {
                for (cl_long i = treeDecomp.bags[0].solution[a].minId; i < treeDecomp.bags[0].solution[a].maxId; i++) {
                    if (treeDecomp.bags[0].solution[a].elements != nullptr) {
                        if (solutionType == TREE) {
                            sols = sols + getCount(i, treeDecomp.bags[0].solution[a].elements, treeDecomp.bags[0].variables.size());
                        } else if (solutionType == ARRAY) {
                            sols = sols + *reinterpret_cast <cl_double *>(&treeDecomp.bags[0].solution[a].elements[i - treeDecomp.bags[0].solution[a].minId]);
                        }
                    }
                }
                if (treeDecomp.bags[0].solution[a].elements != NULL)
                    free(treeDecomp.bags[0].solution[a].elements);
            }

            if (!noExp) {
                boost::multiprecision::cpp_bin_float_100 base = 2, exponent = treeDecomp.bags[0].correction;
                sols = sols * pow(base, exponent);
            }

            if (weighted) {
                sols = sols * tdParser.defaultWeight;
            }

            char buf[128];

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
        if (sols > 0) {
            exit(10);
        } else {
            exit(20);
        }
    }
    catch (cl::Error &err) {
        std::cerr << "\nERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
        if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
            std::string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cout << "Program Info: " << str << std::endl;
        }
    }
}

