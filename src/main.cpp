#include <gpusatutils.h>
#include <satparser.h>
#include <treeparser.h>
#include <solver.h>
#include <sstream>
#include <getopt.h>
#include <fstream>
#include <iostream>

int main(int argc, char *argv[]) {
    std::stringbuf treeD, sat;
    std::string inputLine;
    bool file = false, formula = false;
    int opt;
    while ((opt = getopt(argc, argv, "f:s:")) != -1) {
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
        fprintf(stderr, "Usage: %s [-f treedecomp] -s formula \n", argv[0]);
        exit(EXIT_FAILURE);
    }

    treedecType treeDecomp = parseTreeDecomp(treeD.str());
    satformulaType satFormula = parseSatFormula(sat.str());
    printTreeD(treeDecomp);
    printFormula(satFormula);

    //GPU Code
    std::vector<cl::Platform> my_platforms;
    cl::Context my_context;
    std::vector<cl::Device> my_devices;
    cl::CommandQueue my_queue;
    cl::Program my_program;
    cl::Kernel my_kernel;

    cl::Platform::get(&my_platforms);
    std::vector<cl::Platform>::iterator iter;
    for (iter = my_platforms.begin(); iter != my_platforms.end(); ++iter) {
        if (!strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(),
                    "Advanced Micro Devices, Inc.")) {
            break;
        }
    }
    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,
                                    (cl_context_properties) (*iter)(), 0};
    my_context = cl::Context(CL_DEVICE_TYPE_GPU, cps);
    my_devices = my_context.getInfo<CL_CONTEXT_DEVICES>();
    my_queue = cl::CommandQueue(my_context, my_devices[0]);
    std::string kernelStr = readFile("./kernel/kernel.cl");
    cl::Program::Sources sources(1, std::make_pair(kernelStr.c_str(),
                                                   kernelStr.length()));
    my_program = cl::Program(my_context, sources);
    my_program.build(my_devices);

    solveProblem(treeDecomp, satFormula, treeDecomp.bags[0], my_context, my_kernel, my_program, my_queue);
    int solutions = 0;
    for (int i = 0; i < treeDecomp.bags[0].numSol; i++) {
        solutions += treeDecomp.bags[0].solution[(treeDecomp.bags[0].numv + 1) * i + treeDecomp.bags[0].numv];
    }
    std::cout << "Solutions: " << solutions;
    printSolutions(treeDecomp);
    int test = 0;
}
