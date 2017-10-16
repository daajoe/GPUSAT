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

using namespace gpusat;

long long int getTime();

int main(int argc, char *argv[]) {
    long long int time_total = getTime();
    std::stringbuf treeD, sat;
    std::string inputLine;
    bool file = false, formula = false;
    int opt, combineWidth = 12, maxBag = 22;
    std::string kernelPath = "./kernel/";
    graphTypes graph = INCIDENCE;
    static struct option flags[] = {
            {"formula",       required_argument, 0, 's'},
            {"decomposition", required_argument, 0, 'f'},
            {"combineWidth",  required_argument, 0, 'w'},
            {"maxBagSize",    required_argument, 0, 'm'},
            {"kernelDir",     required_argument, 0, 'c'},
            {"help",          no_argument,       0, 'h'},
            {0,               0,                 0, 0}
    };
    while ((opt = getopt_long(argc, argv, "f:s:c:w:m:h", flags, NULL)) != -1) {
        switch (opt) {
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
            case 'h': {
                std::cout << "Usage: \n" << argv[0] << "\n"
                          << "    --decomposition,-f <treedecomp> : <treedecomp> path to the file containing the tree decomposition\n"
                          << "    --formula,-s <formula>          : <formula> path to the file containing the sat formula\n"
                          << "    --combineWidth,-w <width>       : <width> maximum width to combine bags of the decomposition\n"
                          << "    --maxBagSize,-m <size>          : <size> maximum size of a bag before splitting it\n"
                          << "    --kernelDir,-c <kerneldir>      : <kerneldir> path to the directory containing the kernel files\n"
                          << "    --help,-h                       : prints this message\n";
                exit(EXIT_SUCCESS);
            }
            default:
                std::cerr << "Error: Unknown option\n" << "Usage: \n" << argv[0] << "\n"
                          << "    --decomposition,-f <treedecomp> : <treedecomp> path to the file containing the tree decomposition\n"
                          << "    --formula,-s <formula>          : <formula> path to the file containing the sat formula\n"
                          << "    --combineWidth,-w <width>       : <width> maximum width to combine bags of the decomposition\n"
                          << "    --maxBagSize,-m <size>          : <size> maximum size of a bag before splitting it\n"
                          << "    --kernelDir,-c <kerneldir>      : <kerneldir> path to the directory containing the kernel files\n"
                          << "    --help,-h                       : prints this message\n";
                exit(EXIT_FAILURE);
        }
    }

    // no file flag
    if (!file) {
        while (getline(std::cin, inputLine)) {
            treeD.sputn(inputLine.c_str(), inputLine.size());
            treeD.sputn("\n", 1);
        }
    }

    // error no sat formula given
    if (!formula) {
        std::cerr << "Error: No SAT formula\n" << "Usage: \n" << argv[0] << "\n"
                  << "    --decomposition,-f <treedecomp> : <treedecomp> path to the file containing the tree decomposition\n"
                  << "    --formula,-s <formula>          : <formula> path to the file containing the sat formula\n"
                  << "    --combineWidth,-w <width>       : <width> maximum width to combine bags of the decomposition\n"
                  << "    --maxBagSize,-m <size>          : <size> maximum size of a bag before splitting it\n"
                  << "    --kernelDir,-c <kerneldir>      : <kerneldir> path to the directory containing the kernel files\n"
                  << "    --help,-h                       : prints this message\n";
        exit(EXIT_FAILURE);
    }

    long long int time_parsing = getTime();
    CNFParser cnfParser;
    TDParser tdParser(combineWidth);
    satformulaType satFormula = cnfParser.parseSatFormula(sat.str());
    treedecType treeDecomp = tdParser.parseTreeDecomp(treeD.str());
    if (satFormula.numVars + satFormula.numclauses == treeDecomp.numVars) {
        graph = INCIDENCE;
    } else if (satFormula.numVars == treeDecomp.numVars) {
        graph = PRIMAL;
    } else {
        std::cerr << "Error: Unknown graph type\n" << "Usage: \n" << argv[0] << "\n"
                  << "    --decomposition,-f <treedecomp> : <treedecomp> path to the file containing the tree decomposition\n"
                  << "    --formula,-s <formula>          : <formula> path to the file containing the sat formula\n"
                  << "    --combineWidth,-w <width>       : <width> maximum width to combine bags of the decomposition\n"
                  << "    --maxBagSize,-m <size>          : <size> maximum size of a bag before splitting it\n"
                  << "    --kernelDir,-c <kerneldir>      : <kerneldir> path to the directory containing the kernel files\n"
                  << "    --help,-h                       : prints this message\n";
        exit(EXIT_FAILURE);
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
            if (strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "Advanced Micro Devices, Inc.") == 0) {
                break;
            }
        }
        cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties) (*iter)(), 0};
        context = cl::Context(CL_DEVICE_TYPE_GPU, cps);
        devices = context.getInfo<CL_CONTEXT_DEVICES>();
        queue = cl::CommandQueue(context, devices[0]);
        time_init_opencl = getTime() - time_init_opencl;

        long long int time_build_kernel = getTime();
        struct stat buffer;
        std::string binPath;

#ifdef sType_Double
        switch (graph) {
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
                case PRIMAL:
                    sourcePath = kernelPath + "SAT_d_primal.cl";
                    break;
                case INCIDENCE:
                    sourcePath = kernelPath + "SAT_d_inci.cl";
                    break;
            }
#else
            switch (graph) {
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
        switch (graph) {
            case PRIMAL:
                sol = new Solver_Primal(platforms, context, devices, queue, program, kernel, maxBag, false);
                break;
            case INCIDENCE:
                sol = new Solver_Incidence(platforms, context, devices, queue, program, kernel, maxBag, true);
                break;
        }
        long long int time_solving = getTime();
        (*sol).solveProblem(treeDecomp, satFormula, treeDecomp.bags[0]);
        time_solving = getTime() - time_solving;

        long long int time_model = getTime();
        solType solutions = 0.0;
        if ((*sol).isSat > 0) {
#ifdef sType_Double
            for (cl_long i = 0; i < treeDecomp.bags[0].numSol; i++) {
                if (graph == INCIDENCE) {
                    bool sat = true;
                    int b = 0, c = 0;
                    for (b = 0; treeDecomp.bags[0].variables[b] <= satFormula.numVars && b < treeDecomp.bags[0].numVars; b++) {
                    };
                    for (c = 0; b + c < treeDecomp.bags[0].numVars; c++) {
                        if (((i >> c) & 1) == 0) {
                            sat = false;
                        }
                    }
                    if (sat) {
                        solutions = solutions + treeDecomp.bags[0].solution[i];
                    }
                } else {
                    solutions += treeDecomp.bags[0].solution[i];
                }
            }
            std::cout.precision(17);
            std::cout << "{\n    \"Model Count\": " << solutions;
#else
            for (cl_long i = 0; i < treeDecomp.bags[0].numSol; i++) {
                if (graph == INCIDENCE) {
                    bool sat = true;
                    int b = 0, c = 0;
                    for (b = 0; treeDecomp.bags[0].variables[b] <= satFormula.numVars && b < treeDecomp.bags[0].numVars; b++) {
                    };
                    for (c = 0; b + c < treeDecomp.bags[0].numVars; c++) {
                        if (((i >> c) & 1) == 0) {
                            sat = false;
                        }
                    }
                    if (sat) {
                        solutions = d4_add(solutions, treeDecomp.bags[0].solution[i]);
                    }
                } else {
                    solutions = d4_add(solutions, treeDecomp.bags[0].solution[i]);
                }
            }
            std::cout << "{\n    \"Model Count\": " << d4_to_string(solutions);
#endif

        } else {
            std::cout << "{\n    \"Model Count\": " << 0;
        }
        delete[] treeDecomp.bags[0].solution;
        delete[] treeDecomp.bags[0].variables;
        delete[] satFormula.clauses;
        time_model = getTime() - time_model;
        time_total = getTime() - time_total;
        std::cout << "\n    ,\"Time\":{";
        std::cout << "\n        \"Solving\": " << ((float) time_solving) / 1000;
        std::cout << "\n        ,\"Parsing\": " << ((float) time_parsing) / 1000;
        std::cout << "\n        ,\"Build_Kernel\": " << ((float) time_build_kernel) / 1000;
        std::cout << "\n        ,\"Generate_Model\": " << ((float) time_model) / 1000;
        std::cout << "\n        ,\"Init_OpenCL\": " << ((float) time_init_opencl) / 1000;
        std::cout << "\n        ,\"Total\": " << ((float) time_total) / 1000;
        std::cout << "\n    }";
        std::cout << "\n}";
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