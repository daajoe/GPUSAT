#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <sstream>
#include <fstream>
#include <getopt.h>
#include <regex>
#include <math.h>
#include <gpusatparser.h>
#include <gpusautils.h>
#include <main.h>

std::vector<cl::Platform> platforms;
cl::Context context;
std::vector<cl::Device> devices;
cl::CommandQueue queue;
cl::Program program;
cl::Kernel kernel;

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
    //printTreeD(treeDecomp);
    //printFormula(satFormula);

    try {
        cl::Platform::get(&platforms);
        std::vector<cl::Platform>::iterator iter;
        for (iter = platforms.begin(); iter != platforms.end(); ++iter) {
            if (!strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(),
                        "Advanced Micro Devices, Inc.")) {
                break;
            }
        }
        cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties) (*iter)(), 0};
        context = cl::Context(CL_DEVICE_TYPE_GPU, cps);
        devices = context.getInfo<CL_CONTEXT_DEVICES>();
        queue = cl::CommandQueue(context, devices[0]);
        std::string kernelStr = readFile("./kernel/kernel.cl");
        cl::Program::Sources sources(1, std::make_pair(kernelStr.c_str(),
                                                       kernelStr.length()));
        program = cl::Program(context, sources);
        program.build(devices);
        solveProblem(treeDecomp, satFormula, treeDecomp.bags[0]);
        int solutions = 0;
        for (int i = 0; i < treeDecomp.bags[0].numSol; i++) {
            solutions += treeDecomp.bags[0].solution[i * 2 + 1];
        }
        std::cout << "Solutions: " << solutions;
        if (solutions > 0) {
            cl_int *solution = new cl_int[satFormula.numVar]();
            //genSolution(treeDecomp, solution, treeDecomp.bags[0]);
            /*std::cout << "\nSolution: ";
            for (int i = 0; i < satFormula.numVar; i++)
                std::cout << " " << solution[i];*/
        }
        printSolutions(treeDecomp);

    }
    catch (cl::Error err) {
        std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" <<
                  std::endl;
        if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
            std::string str =
                    program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cout << "Program Info: " << str << std::endl;
        }
    }
    catch (std::string msg) {
        std::cerr << "Exception caught in main(): " << msg << std::endl;
    }
}

void
solveProblem(treedecType decomp, satformulaType formula, bagType node) {

    for (int i = 0; i < node.nume; i++) {
        cl_int edge = node.edges[i] - 1;
        solveProblem(decomp, formula, decomp.bags[edge]);
    }
    if (node.nume > 1) {
        //join node
        solveJoin(decomp, node);
    } else if (node.nume == 1 && decomp.bags[node.edges[0] - 1].numVars > node.numVars) {
        //forget node
        solveForget(decomp, node);
    } else {
        if (node.nume == 0) {
            //leaf node
            solveLeaf(formula, node);
        } else if (node.nume == 1 && decomp.bags[node.edges[0] - 1].numVars < node.numVars) {
            //introduce node
            solveIntroduce(decomp, formula, node);
        }
    }
}

void solveIntroduce(treedecType &decomp, satformulaType &formula, bagType &node) {
    cl::Buffer bufSol(context,
                      CL_MEM_READ_WRITE,
                      sizeof(cl_int) * (node.numSol * 2));
    cl::Buffer bufVertices(context,
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_int) * (node.numVars),
                           node.variables);
    cl::Buffer bufClauses(context,
                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_int) * (formula.totalNumVar),
                          formula.clauses);
    cl::Buffer bufNumVarsC(context,
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_int) * (formula.numclauses),
                           formula.numVarsC);
    cl::Buffer bufSolNext(context,
                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_int) * (decomp.bags[node.edges[0] - 1].numSol * 2),
                          decomp.bags[node.edges[0] - 1].solution);
    cl::Buffer bufNextVars(context,
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_int) * (decomp.bags[node.edges[0] - 1].numVars),
                           decomp.bags[node.edges[0] - 1].variables);
    kernel = cl::Kernel(program, "solveIntroduce");
    kernel.setArg(0, bufClauses);
    kernel.setArg(1, bufNumVarsC);
    kernel.setArg(2, formula.numclauses);
    kernel.setArg(3, bufSol);
    kernel.setArg(4, node.numVars);
    kernel.setArg(5, bufSolNext);
    kernel.setArg(6, decomp.bags[node.edges[0] - 1].numVars);
    kernel.setArg(7, bufVertices);
    kernel.setArg(8, bufNextVars);
    size_t numKernels = node.numSol;
    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(numKernels));
    queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_int) * (node.numSol * 2),
                            node.solution);
}

void solveLeaf(satformulaType &formula, bagType &node) {
    cl::Buffer bufSol(context,
                      CL_MEM_READ_WRITE,
                      sizeof(cl_int) * (node.numSol * 2));
    cl::Buffer bufVertices(context,
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_int) * (node.numVars),
                           node.variables);
    cl::Buffer bufClauses(context,
                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_int) * (formula.totalNumVar),
                          formula.clauses);
    cl::Buffer bufNumVarsC(context,
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_int) * (formula.numclauses),
                           formula.numVarsC);
    kernel = cl::Kernel(program, "solveLeaf");
    kernel.setArg(0, bufClauses);
    kernel.setArg(1, bufNumVarsC);
    kernel.setArg(2, formula.numclauses);
    kernel.setArg(3, bufSol);
    kernel.setArg(4, node.numVars);
    kernel.setArg(5, bufVertices);
    size_t numKernels = node.numSol;
    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(numKernels));
    queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_int) * (node.numSol * 2),
                            node.solution);
}

void solveForget(treedecType &decomp, bagType &node) {
    cl::Buffer bufSol(context,
                      CL_MEM_READ_WRITE,
                      sizeof(cl_int) * (node.numSol * 2));
    cl::Buffer bufVertices(context,
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_int) * (node.numVars),
                           node.variables);
    cl::Buffer bufNextSol(context,
                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_int) * (decomp.bags[node.edges[0] - 1].numSol * 2),
                          decomp.bags[node.edges[0] - 1].solution);
    cl::Buffer bufSolVars(context,
                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_int) * node.numVars,
                          node.variables);
    cl::Buffer bufNextVars(context,
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_int) * decomp.bags[node.edges[0] - 1].numVars,
                           decomp.bags[node.edges[0] - 1].variables);
    kernel = cl::Kernel(program, "solveForget");
    kernel.setArg(0, bufSol);
    kernel.setArg(1, node.numVars);
    kernel.setArg(2, bufSolVars);
    kernel.setArg(3, bufNextSol);
    kernel.setArg(4, decomp.bags[node.edges[0] - 1].numVars);
    kernel.setArg(5, bufNextVars);
    size_t numKernels = decomp.bags[node.edges[0] - 1].numSol;
    queue.enqueueFillBuffer(bufSol, 0, 0, sizeof(cl_int) * (node.numSol * 2));
    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(numKernels));
    queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_int) * (node.numSol * 2), node.solution);
}

void solveJoin(treedecType &decomp, bagType &node) {
    cl::Buffer bufSol(context,
                      CL_MEM_READ_WRITE,
                      sizeof(cl_int) * (node.numSol * 2));
    cl::Buffer bufVertices(context,
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_int) * (node.numVars),
                           node.variables);
    cl_int *solutions = new cl_int[node.numSol * 4];
    std::copy(decomp.bags[node.edges[0] - 1].solution, decomp.bags[node.edges[0] - 1].solution + node.numSol * 2,
              solutions);
    std::copy(decomp.bags[node.edges[1] - 1].solution, decomp.bags[node.edges[1] - 1].solution + node.numSol * 2,
              &solutions[node.numSol * 2]);
    cl::Buffer bufSolOther(context,
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_int) * (node.numSol * 4),
                           solutions);
    kernel = cl::Kernel(program, "solveJoin");
    kernel.setArg(0, bufSol);
    kernel.setArg(1, bufSolOther);
    kernel.setArg(2, node.numSol);
    size_t numKernels = node.numSol;
    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(numKernels));
    queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_int) * (node.numSol * 2), node.solution);
}

void genSolution(treedecType decomp, cl_int *solution, bagType node) {
    for (int i = 0; i < node.numSol; i++) {
        int a = 0;
        for (a = 0; a < node.numVars; a++) {
            if (solution[abs(node.solution[i * (node.numVars + 1) + a])] != 0 &&
                node.solution[i * (node.numVars + 1) + a] != solution[abs(node.solution[i * (node.numVars + 1) + a])]) {
                break;
            }
        }
        if (a == node.numVars) {
            for (a = 0; a < node.numVars; a++) {
                solution[abs(node.solution[i * (node.numVars + 1) + a])] = node.solution[i * (node.numVars + 1) + a];
            }
        }
    }
    for (int i = 0; i < node.nume; i++) {
        genSolution(decomp, solution, decomp.bags[node.edges[i] - 1]);
    }
}
