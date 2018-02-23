#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <sstream>
#include <fstream>
#include <getopt.h>
#include <regex>
#include <math.h>
#include <chrono>
#include <types.h>
#include <gpusatparser.h>
#include <gpusautils.h>
#include <sys/stat.h>
#include <numeric>
#include <solver.h>
#include <d4_utils.h>
//#include <quadmath.h>
#include <stdlib.h>
#include <stdio.h>

extern "C" {
#include "quadmath.h"
}

using namespace gpusat;

long long int getTime();

void printUsage(char *argv[]) {
    std::cout << "Usage: \n" << argv[0] << "-f <decomp path> -s <formula path> [-w <width>] [-c <kernel path>] [-g <1|2|3>] [--CPU] [--weighted] [-h]\n"
              << "    --decomposition,-f <treedecomp> : <treedecomp> path to the file containing the tree decomposition\n"
              << "    --formula,-s <formula>          : <formula> path to the file containing the sat formula\n"
              << "    --combineWidth,-w <width>       : <width> maximum width to combine bags of the decomposition\n"
              << "    --kernelDir,-c <dir>            : directory containing the kernel files\n"
              << "    --graph,-g <1|2|3>              : indicator for the given graph type (1=PRIMAL, 2=INCIDENCE, 3=DUAL)\n"
              << "    --CPU                           : uses the CPU instead of the GPU\n"
              << "    --weighted                      : calculates the weighted model count of the formula\n"
              << "    --help,-h                       : prints this message\n";
}

int main(int argc, char *argv[]) {
    long long int time_total = getTime();
    std::stringbuf treeD, sat;
    std::string inputLine;
    bool file = false, formula = false;
    int opt, combineWidth = 12, maxBag = 22;
    std::string kernelPath = "./kernel/";
    graphTypes graph = NONE;
    cl_long getStats = 0;
    bool factR = true;
    bool cpu = false;
    bool weighted = false;
    static struct option flags[] = {
            {"formula",       required_argument, 0, 's'},
            {"decomposition", required_argument, 0, 'f'},
            {"combineWidth",  required_argument, 0, 'w'},
            {"maxBagSize",    required_argument, 0, 'm'},
            {"kernelDir",     required_argument, 0, 'c'},
            {"graph",         required_argument, 0, 'g'},
            {"help",          no_argument,       0, 'h'},
            {"getStats",      no_argument,       0, 'a'},
            {"noFactRemoval", no_argument,       0, 'b'},
            {"CPU",           no_argument,       0, 'd'},
            {"weighted",      no_argument,       0, 'e'},
            {0,               0,                 0, 0}
    };
    //parse flags
    while ((opt = getopt_long(argc, argv, "f:s:c:w:m:ahg:", flags, NULL)) != -1) {
        switch (opt) {
            case 'e': {
                weighted = true;
            }
            case 'f': {
                // input tree decomposition file
                file = true;
                std::ifstream fileIn(optarg);
                while (getline(fileIn, inputLine)) {
                    treeD.sputn(inputLine.c_str(), inputLine.size());
                    treeD.sputn("\n", 1);
                }
                break;
            }
            case 'b': {
                factR = false;
            }
            case 's': {
                // input sat formula
                formula = true;
                std::ifstream fileIn(optarg);
                while (getline(fileIn, inputLine)) {
                    sat.sputn(inputLine.c_str(), inputLine.size());
                    sat.sputn("\n", 1);
                }
                break;
            }
            case 'c': {
                kernelPath = std::string(optarg);
                break;
            }
            case 'w': {
                combineWidth = std::atoi(optarg);
                break;
            }
            case 'm': {
                maxBag = std::atoi(optarg);
                break;
            }
            case 'g': {
                int type = std::atoi(optarg);
                switch (type) {
                    case 1: {
                        graph = PRIMAL;
                        break;
                    }
                    case 2: {
                        graph = INCIDENCE;
                        break;
                    }
                    case 3: {
                        graph = DUAL;
                        break;
                    }
                    default: {
                        std::cerr << "Error: Unknown graph type\n";
                        printUsage(argv);
                        exit(EXIT_FAILURE);
                    }
                }
                break;
            }
            case 'a': {
                getStats = 1;
                break;
            }
            case 'd': {
                cpu = true;
                break;
            }
            case 'h': {
                printUsage(argv);
                exit(EXIT_SUCCESS);
            }
            default:
                std::cerr << "Error: Unknown option\n";
                printUsage(argv);
                exit(EXIT_FAILURE);
        }
    }

    // no decomposition file flag
    if (!file) {
        while (getline(std::cin, inputLine)) {
            treeD.sputn(inputLine.c_str(), inputLine.size());
            treeD.sputn("\n", 1);
        }
    }

    // error no sat formula given
    if (!formula) {
        std::cerr << "Error: No SAT formula\n";
        printUsage(argv);
        exit(EXIT_FAILURE);
    }

    long long int time_parsing = getTime();
    CNFParser cnfParser(weighted);
    TDParser tdParser(combineWidth, factR, maxBag);
    std::string satString = sat.str();
    std::string treeDString = treeD.str();
    if (satString.size() < 10) {
        std::cerr << "Error: SAT formula\n";
        exit(EXIT_FAILURE);
    }
    if (treeDString.size() < 9) {
        std::cerr << "Error: tree decomposition\n";
        exit(EXIT_FAILURE);
    }
    //parse the sat formula
    satformulaType satFormula = cnfParser.parseSatFormula(sat.str());
    //parse the tree decomposition
    treedecType treeDecomp = tdParser.parseTreeDecomp(treeD.str(), satFormula);
    //found unsat during preprocessing
    if (satFormula.unsat) {
        std::cout << "{\n    \"Model Count\": " << 0;
        time_total = getTime() - time_total;
        std::cout << "\n    ,\"Time\":{";
        std::cout << "\n        \"Solving\": " << 0;
        std::cout << "\n        ,\"Parsing\": " << ((float) time_parsing) / 1000;
        std::cout << "\n        ,\"Build_Kernel\": " << 0;
        std::cout << "\n        ,\"Generate_Model\": " << 0;
        std::cout << "\n        ,\"Init_OpenCL\": " << 0;
        std::cout << "\n        ,\"Total\": " << ((float) time_total) / 1000;
        std::cout << "\n    }";
        std::cout << "\n    ,\"Statistics\":{";
        std::cout << "\n        \"Num Join\": " << 0;
        std::cout << "\n        ,\"Num Forget\": " << 0;
        std::cout << "\n        ,\"Num Introduce\": " << 0;
        std::cout << "\n        ,\"Num Leaf\": " << 0;

        if (getStats) {
            std::cout << "\n        ,\"Actual Paths\":{";
            std::cout << "\n            \"min\": " << 0;
            std::cout << "\n            ,\"max\": " << 0;
            std::cout << "\n            ,\"mean\": " << 0;
            std::cout << "\n        }";
            std::cout << "\n        ,\"Current Paths\":{";
            std::cout << "\n            \"min\": " << 0;
            std::cout << "\n            ,\"max\": " << 0;
            std::cout << "\n            ,\"mean\": " << 0;
            std::cout << "\n        }";
        }
        std::cout << "\n    }";
        std::cout << "\n}";
        exit(20);

    }
    if (graph == NONE) {
        if (satFormula.clauses.size() == treeDecomp.numVars) {
            // decomposition is of the dual
            graph = DUAL;
        } else if (satFormula.numVars == treeDecomp.numVars) {
            // decomposition is of the incidence graph
            graph = PRIMAL;
        } else if (satFormula.numVars + satFormula.clauses.size() == treeDecomp.numVars) {
            // decomposition is of the primal graph
            graph = INCIDENCE;
        } else {
            std::cerr << "Error: Unknown graph type\n";
            printUsage(argv);
            exit(EXIT_FAILURE);
        }
    }
    time_parsing = getTime() - time_parsing;

    std::vector<cl::Platform> platforms;
    cl::Context context;
    std::vector<cl::Device> devices;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Kernel kernel;

    try {
        long long int time_init_opencl = getTime();
        cl::Platform::get(&platforms);
        std::vector<cl::Platform>::iterator iter;
        for (iter = platforms.begin(); iter != platforms.end(); ++iter) {

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
                break;
            }
        }
        if (iter == platforms.end()) {
            std::cout << "ERROR: no GPU found!";
            exit(1);
        }
        time_init_opencl = getTime() - time_init_opencl;

        long long int time_build_kernel = getTime();
        struct stat buffer;
        std::string binPath;

#ifdef sType_Double
        switch (graph) {
            case DUAL:
                binPath = kernelPath + "SAT_d_d.clbin";
                break;
            case PRIMAL:
                binPath = kernelPath + "SAT_d_p.clbin";
                break;
            case INCIDENCE:
                binPath = kernelPath + "SAT_d_i.clbin";
                break;
            default:
                break;
        }
#else
        switch (graph) {
            case DUAL:
                binPath = kernelPath + "SAT_d4_d.clbin";
                break;
            case PRIMAL:
                binPath = kernelPath + "SAT_d4_p.clbin";
                break;
            case INCIDENCE:
                binPath = kernelPath + "SAT_d4_i.clbin";
                break;
        }
#endif

#ifndef DEBUG
        if (stat(binPath.c_str(), &buffer) != 0) {
#endif
            //create kernel binary if it doesn't exist
            std::string sourcePath;

#ifdef sType_Double
            switch (graph) {
                case DUAL:
                    sourcePath = kernelPath + "SAT_d_dual.cl";
                    break;
                case PRIMAL:
                    sourcePath = kernelPath + "SAT_d_primal.cl";
                    break;
                case INCIDENCE:
                    sourcePath = kernelPath + "SAT_d_inci.cl";
                    break;
            }
#else
            switch (graph) {
                case DUAL:
                    sourcePath = kernelPath + "SAT_d4_dual.cl";
                    break;
                case PRIMAL:
                    sourcePath = kernelPath + "SAT_d4_primal.cl";
                    break;
                case INCIDENCE:
                    sourcePath = kernelPath + "SAT_d4_inci.cl";
                    break;
            }
#endif
            std::string kernelStr = GPUSATUtils::readFile(sourcePath);
            cl::Program::Sources sources(1, std::make_pair(kernelStr.c_str(), kernelStr.length()));
            program = cl::Program(context, sources);
            program.build(devices);

            const std::vector<size_t> binSizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
            std::vector<char> binData((unsigned long long int) std::accumulate(binSizes.begin(), binSizes.end(), 0));
            char *binChunk = &binData[0];

            std::vector<char *> binaries;
            for (const size_t &binSize : binSizes) {
                binaries.push_back(binChunk);
                binChunk += binSize;
            }

            program.getInfo(CL_PROGRAM_BINARIES, &binaries[0]);
            std::ofstream binaryfile(binPath.c_str(), std::ios::binary);
            for (unsigned int i = 0; i < binaries.size(); ++i)
                binaryfile.write(binaries[i], binSizes[i]);
            binaryfile.close();
#ifndef DEBUG
        } else {
            //load kernel binary

            long size = 0;
            cl_int err;
            std::string kernelStr = GPUSATUtils::readBinary(binPath);
            cl::Program::Binaries bins(1, std::make_pair((const void *) kernelStr.data(), kernelStr.size()));
            program = cl::Program(context, devices, bins, nullptr, &err);
            program.build(devices);
        }
#endif
        time_build_kernel = getTime() - time_build_kernel;

        Solver *sol;
        bagType next;
        switch (graph) {
            case PRIMAL:
                sol = new Solver_Primal(platforms, context, devices, queue, program, kernel, maxBag, false, getStats);
                next.numSol = pow(2, next.variables.size());
                next.variables.assign(treeDecomp.bags[0].variables.begin(),
                                      treeDecomp.bags[0].variables.begin() + std::min((cl_long) treeDecomp.bags[0].variables.size(), (cl_long) 12));
                break;
            case DUAL:
                sol = new Solver_Dual(platforms, context, devices, queue, program, kernel, maxBag, false, getStats);
                next.numSol = pow(2, next.variables.size());
                next.variables.assign(treeDecomp.bags[0].variables.begin(),
                                      treeDecomp.bags[0].variables.begin() + std::min((cl_long) treeDecomp.bags[0].variables.size(), (cl_long) 12));
                break;
            case INCIDENCE:
                sol = new Solver_Incidence(platforms, context, devices, queue, program, kernel, maxBag, true, getStats);
                std::vector<cl_long> *vars = new std::vector<cl_long>;
                for (int i = 0; i < treeDecomp.bags[0].variables.size(); ++i) {
                    if (treeDecomp.bags[0].variables[i] <= satFormula.numVars) {
                        (*vars).push_back(treeDecomp.bags[0].variables[i]);
                    } else {
                        break;
                    }
                }
                next.numSol = pow(2, next.variables.size());
                next.variables = *vars;
                break;
        }
        long long int time_solving = getTime();
        (*sol).solveProblem(treeDecomp, satFormula, treeDecomp.bags[0], next);
        time_solving = getTime() - time_solving;

        //sum up last node solutions
        long long int time_model = getTime();
        __float128 solutions = 0.0;
        if ((*sol).isSat > 0) {
            cl_long bagSizeNode = static_cast<cl_long>(pow(2, std::min((cl_long) maxBag, (cl_long) treeDecomp.bags[0].variables.size())));
            if (graph == DUAL) {
                for (cl_long a = 0; a < treeDecomp.bags[0].numSol / bagSizeNode; a++) {
                    if (treeDecomp.bags[0].solution[a] == nullptr) {
                        continue;
                    }
                    for (cl_long i = 0; i < bagSizeNode; i++) {
                        solutions = solutions + treeDecomp.bags[0].solution[a][i] * (1 - ((__builtin_popcount(i + a * bagSizeNode) % 2 == 1) * 2));
                    }
                }
            } else {
                for (cl_long a = 0; a < treeDecomp.bags[0].numSol / bagSizeNode; a++) {
                    if (treeDecomp.bags[0].solution[a] == nullptr) {
                        continue;
                    }
                    for (cl_long i = 0; i < bagSizeNode; i++) {
                        solutions = solutions + treeDecomp.bags[0].solution[a][i];
                    }
                }
            }
            if (!weighted && graph != DUAL) {
                __float128 base = 0.78, exponent = satFormula.numVars;
                solutions = solutions / powq(base, exponent);
            } else if (graph != DUAL) {
                solutions = solutions * tdParser.defaultWeight;
            }

            if (graph == DUAL) {
                std::set<cl_long> varSet;
                for (int j = 0; j < satFormula.clauses.size(); ++j) {
                    for (int i = 0; i < satFormula.clauses[j].size(); ++i) {
                        varSet.insert(satFormula.clauses[j][i]);
                    }
                }
                for (int i = 1; i <= satFormula.numVars; ++i) {
                    if (varSet.find(i) == varSet.end() && varSet.find(-i) == varSet.end())
                        solutions *= 2;
                }
            }
            char buf[128];
            quadmath_snprintf(buf, sizeof buf, "%.30Qe", solutions);
            std::cout << "{\n    \"Model Count\": " << buf;

        } else {
            std::cout << "{\n    \"Model Count\": " << 0;
        }
        time_model = getTime() - time_model;
        time_total = getTime() - time_total;
        std::sort(sol->numSolPaths.begin(), sol->numSolPaths.end());
        std::sort(sol->numHoldPaths.begin(), sol->numHoldPaths.end());
        std::cout.precision(6);
        std::cout << "\n    ,\"Time\":{";
        std::cout << "\n        \"Solving\": " << ((float) time_solving) / 1000;
        std::cout << "\n        ,\"Parsing\": " << ((float) time_parsing) / 1000;
        std::cout << "\n        ,\"Build_Kernel\": " << ((float) time_build_kernel) / 1000;
        std::cout << "\n        ,\"Generate_Model\": " << ((float) time_model) / 1000;
        std::cout << "\n        ,\"Init_OpenCL\": " << ((float) time_init_opencl) / 1000;
        std::cout << "\n        ,\"Total\": " << ((float) time_total) / 1000;
        std::cout << "\n    }";
        std::cout << "\n    ,\"Statistics\":{";
        std::cout << "\n        \"Num Join\": " << sol->numJoin;
        std::cout << "\n        ,\"Num Forget\": " << sol->numForget;
        std::cout << "\n        ,\"Num Introduce\": " << sol->numIntroduce;
        std::cout << "\n        ,\"Num Leaf\": " << sol->numLeafs;

        if (getStats) {
            std::cout << "\n        ,\"Actual Paths\":{";
            std::cout << "\n            \"min\": " << sol->numSolPaths[0];
            std::cout << "\n            ,\"max\": " << sol->numSolPaths[sol->numSolPaths.size() - 1];
            std::cout << "\n            ,\"mean\": " << sol->numSolPaths[sol->numSolPaths.size() / 2];
            std::cout << "\n        }";
            std::cout << "\n        ,\"Current Paths\":{";
            std::cout << "\n            \"min\": " << sol->numHoldPaths[0];
            std::cout << "\n            ,\"max\": " << sol->numHoldPaths[sol->numHoldPaths.size() - 1];
            std::cout << "\n            ,\"mean\": " << sol->numHoldPaths[sol->numHoldPaths.size() / 2];
            std::cout << "\n        }";
        }
        std::cout << "\n    }";
        std::cout << "\n}";
        std::cout.flush();
        if (solutions > 0.0) {
            exit(10);
        } else {
            exit(20);
        }
    }
    catch (cl::Error &err) {
        std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
        if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
            std::string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cout << "Program Info: " << str << std::endl;
        }
    }
    catch (std::string &msg) {
        std::cerr << "Exception caught in main(): " << msg << std::endl;
    }
}

long long int getTime() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}