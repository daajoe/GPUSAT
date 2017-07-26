#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <sstream>
#include <fstream>
#include <getopt.h>
#include <regex>
#include <math.h>
#include <chrono>
#include <gpusatparser.h>
#include <gpusautils.h>
#include <sys/stat.h>
#include <numeric>
#include <cstdlib>
#include <solver.h>
#include <types.h>

using namespace gpusat;

long long int getTime();

int main(int argc, char *argv[]) {
    long long int time_total = getTime();
    std::stringbuf treeD, sat;
    std::string inputLine;
    bool file = false, formula = false;
    int opt;
    std::string kernelPath = "./kernel/";
    while ((opt = getopt(argc, argv, "f:s:c:")) != -1) {
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
            default:
                fprintf(stderr, "Usage: %s [-f treedecomp] -s formula \n", argv[0]);
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
        fprintf(stderr, "Usage: %s [-f treedecomp] -s formula [-c kerneldir] \n", argv[0]);
        exit(EXIT_FAILURE);
    }

    long long int time_parsing = getTime();
    CNFParser cnfParser;
    TDParser tdParser;
    treedecType treeDecomp = tdParser.parseTreeDecomp(treeD.str());
    satformulaType satFormula = cnfParser.parseSatFormula(sat.str());
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
            if (!strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "Advanced Micro Devices, Inc.")) {
                break;
            }
        }
        cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties) (*iter)(), 0};
        context = cl::Context(CL_DEVICE_TYPE_GPU, cps);
        devices = context.getInfo<CL_CONTEXT_DEVICES>();
        queue = cl::CommandQueue(context, devices[0]);
        time_init_opencl = getTime() - time_init_opencl;

        long long int time_build_kernel = getTime();
        struct stat buffer;

        std::string binPath(kernelPath + "SAT.clbin");
        if (stat(binPath.c_str(), &buffer) != 0) {
            //create kernel binary if it doesn't exist

            std::string sourcePath(kernelPath + "SAT.cl");
            std::string kernelStr = GPUSATUtils::readFile(sourcePath.c_str());
            cl::Program::Sources sources(1, std::make_pair(kernelStr.c_str(),
                                                           kernelStr.length()));
            program = cl::Program(context, sources);
            program.build(devices);

            const std::vector<size_t> binSizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
            std::vector<char> binData(
                    (unsigned long long int) std::accumulate(binSizes.begin(), binSizes.end(), 0));
            char *binChunk = &binData[0];

            std::vector<char *> binaries;
            for (unsigned int i = 0; i < binSizes.size(); ++i) {
                binaries.push_back(binChunk);
                binChunk += binSizes[i];
            }

            program.getInfo(CL_PROGRAM_BINARIES, &binaries[0]);
            std::ofstream binaryfile(binPath.c_str(), std::ios::binary);
            for (unsigned int i = 0; i < binaries.size(); ++i)
                binaryfile.write(binaries[i], binSizes[i]);
            binaryfile.close();
        } else {
            //load kernel binary

            long size = 0;
            cl_int err;
            std::string kernelStr = GPUSATUtils::readBinary(binPath.c_str());
            cl::Program::Binaries bins(1, std::make_pair((const void *) kernelStr.data(), kernelStr.size()));
            program = cl::Program(context, devices, bins, NULL, &err);
            program.build(devices);
        }
        time_build_kernel = getTime() - time_build_kernel;

        Solver sol(platforms, context, devices, queue, program, kernel);
        long long int time_solving = getTime();
        sol.solveProblem(treeDecomp, satFormula, treeDecomp.bags[0]);
        time_solving = getTime() - time_solving;

        long long int time_model = getTime();
        if (sol.isSat > 0) {
            solType solutions = 0;
            for (cl_long i = 0; i < treeDecomp.bags[0].numSol; i++) {
                bagType &n = treeDecomp.bags[0];
                solutions += treeDecomp.bags[0].solution[i];
            }
            printf("{\n    \"Model Count\": %e", solutions);
        } else {
            std::cout << "{\n    \"Model Count\": " << 0;
        }
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

    }
    catch (cl::Error
           err) {
        std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" <<
                  std::endl;
        if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
            std::string str =
                    program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cout << "Program Info: " << str << std::endl;
        }
    }
    catch (std::string
           msg) {
        std::cerr << "Exception caught in main(): " << msg << std::endl;
    }
}

long long int getTime() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

namespace gpusat {

}