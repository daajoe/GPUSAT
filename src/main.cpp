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

std::string kernelStr = readFile("../kernel/kernel.cl");/*
        "__kernel void solveLeaf(const __global long *clauses, const __global long *numVarsC, long numclauses,\n"
                "                        __global long *solutions, long numV, const __global long *vertices) {\n"
                "    int i;\n"
                "    long id = get_global_id(0);\n"
                "    for (i = 0; i < numV; i++) {\n"
                "        solutions[(numV + 1) * id + i] = 5;\n"
                "    }\n"
                "}\n";*/
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
    printTreeD(treeDecomp);
    printFormula(satFormula);

    //GPU Code

    try {
        /////////////////////////////////////////////////////////////////
        // Find the platform
        /////////////////////////////////////////////////////////////////
        cl::Platform::get(&platforms);
        std::vector<cl::Platform>::iterator iter;
        for (iter = platforms.begin(); iter != platforms.end(); ++iter) {
            if (!strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(),
                        "Advanced Micro Devices, Inc.")) {
                break;
            }
        }
        /////////////////////////////////////////////////////////////////
        // Create an OpenCL context
        /////////////////////////////////////////////////////////////////
        cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties) (*iter)(), 0};
        context = cl::Context(CL_DEVICE_TYPE_GPU, cps);
        /////////////////////////////////////////////////////////////////
        // Detect OpenCL devices
        /////////////////////////////////////////////////////////////////
        devices = context.getInfo<CL_CONTEXT_DEVICES>();
        /////////////////////////////////////////////////////////////////
        // Create an OpenCL command queue
        /////////////////////////////////////////////////////////////////
        queue = cl::CommandQueue(context, devices[0]);
        /////////////////////////////////////////////////////////////////
        // Create OpenCL memory buffers
        /////////////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////
        // Load CL file, build CL program object, create CL kernel object
        /////////////////////////////////////////////////////////////////
        cl::Program::Sources sources(1, std::make_pair(kernelStr.c_str(),
                                                       kernelStr.length()));
        program = cl::Program(context, sources);
        program.build(devices);
/*
        kernel = cl::Kernel(program, "solveLeaf");
        cl::Buffer bufSol = cl::Buffer(context,
                                       CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                       sizeof(cl_long) * (treeDecomp.bags[7].numSol * (treeDecomp.bags[7].numv + 1)),
                                       treeDecomp.bags[7].solution);
        cl::Buffer bufVertices = cl::Buffer(context,
                                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                            sizeof(cl_long) * (treeDecomp.bags[7].numv),
                                            treeDecomp.bags[7].vertices);
        cl::Buffer bufClauses = cl::Buffer(context,
                                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                           sizeof(cl_long) * (satFormula.totalNumVar),
                                           satFormula.clauses);
        cl::Buffer bufNumVarsC = cl::Buffer(context,
                                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                            sizeof(cl_long) * (satFormula.numclauses),
                                            satFormula.numVarsC);
        /////////////////////////////////////////////////////////////////
        // Set the arguments that will be used for kernel execution
        /////////////////////////////////////////////////////////////////
        kernel.setArg(0, bufClauses);
        kernel.setArg(1, bufNumVarsC);
        kernel.setArg(2, satFormula.numclauses);
        kernel.setArg(3, bufSol);
        kernel.setArg(4, treeDecomp.bags[7].numv);
        kernel.setArg(5, bufVertices);
        /////////////////////////////////////////////////////////////////
        // Enqueue the kernel to the queue
        // with appropriate global and local work sizes
        /////////////////////////////////////////////////////////////////
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(0),
                                   cl::NDRange((size_t) pow(2, treeDecomp.bags[7].numv)));

        /////////////////////////////////////////////////////////////////
        // Enqueue blocking call to read back buffer Y
        /////////////////////////////////////////////////////////////////
        queue.enqueueReadBuffer(bufSol, CL_TRUE, 0,
                                sizeof(cl_long) * (treeDecomp.bags[7].numSol * (treeDecomp.bags[7].numv + 1)),
                                treeDecomp.bags[7].solution);*/
        //solve
        solveProblem(treeDecomp, satFormula, treeDecomp.bags[0]);
        int solutions = 0;
        for (int i = 0; i < treeDecomp.bags[0].numSol; i++) {
            solutions += treeDecomp.bags[0].solution[(treeDecomp.bags[0].numv + 1) * i + treeDecomp.bags[0].numv];
        }
        std::cout << "Solutions: " << solutions;
        printSolutions(treeDecomp);

    }
    catch (cl::Error err) {
        /////////////////////////////////////////////////////////////////
        // Catch OpenCL errors and print log if it is a build error
        /////////////////////////////////////////////////////////////////
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
    int test = 0;
}

void
solveProblem(treedecType decomp, satformulaType formula, bagType node) {

    for (int i = 0; i < node.nume; i++) {
        cl_long edge = node.edges[i] - 1;
        solveProblem(decomp, formula, decomp.bags[edge]);
    }
    cl::Buffer bufSol(context,
                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                      sizeof(cl_long) * (node.numSol * (node.numv + 1)),
                      node.solution);
    cl::Buffer bufVertices(context,
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_long) * (node.numv),
                           node.vertices);
    if (node.nume > 1) {
        //join node
        long size = 0;
        for (int i = 0; i < node.nume; i++) {
            size += decomp.bags[node.edges[i] - 1].numSol * (decomp.bags[node.edges[i] - 1].numv + 1);
        }
        cl_long *solutions = new cl_long[size];
        long s = 0;
        for (int i = 0; i < node.nume; i++) {
            std::copy(decomp.bags[node.edges[i] - 1].solution,
                      decomp.bags[node.edges[i] - 1].solution +
                      decomp.bags[node.edges[i] - 1].numSol * (decomp.bags[node.edges[i] - 1].numv + 1), solutions + s);
            s += decomp.bags[node.edges[i] - 1].numSol * (decomp.bags[node.edges[i] - 1].numv + 1);
        }
        cl::Buffer bufSolOther(context,
                               CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               sizeof(cl_long) * (s),
                               solutions);
        kernel = cl::Kernel(program, "solveJoin");
        kernel.setArg(0, bufSol);
        kernel.setArg(1, node.numv);
        kernel.setArg(2, node.nume);
        kernel.setArg(3, bufVertices);
        kernel.setArg(4, bufSolOther);
        kernel.setArg(5, node.numSol);
        size_t numKernels = pow(2, node.numv);
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(numKernels));
        queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_long) * (node.numSol * (node.numv + 1)), node.solution);

//        solveJoin(node.solution, node.numv, node.nume, node.vertices, solutions, node.numSol);

    } else if (node.nume == 1 && decomp.bags[node.edges[0] - 1].numv > node.numv) {
        //forget node
        cl::Buffer bufSolNext(context,
                              CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(cl_long) * (decomp.bags[node.edges[0] - 1].numSol *
                                                 (decomp.bags[node.edges[0] - 1].numv + 1)),
                              decomp.bags[node.edges[0] - 1].solution);
        kernel = cl::Kernel(program, "solveForget");
        kernel.setArg(0, bufSol);
        kernel.setArg(1, node.numv);
        kernel.setArg(2, bufSolNext);
        kernel.setArg(3, decomp.bags[node.edges[0] - 1].numv);
        kernel.setArg(4, decomp.bags[node.edges[0] - 1].numSol);
        kernel.setArg(5, bufVertices);
        size_t numKernels = pow(2, node.numv);
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(numKernels));
        queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_long) * (node.numSol * (node.numv + 1)), node.solution);

//        solveForget(node.solution, node.numv, decomp.bags[node.edges[0] - 1].solution,
//                    decomp.bags[node.edges[0] - 1].numv, decomp.bags[node.edges[0] - 1].numSol, node.vertices);

    } else {
        if (node.nume == 0) {
            //leaf node
            cl::Buffer bufClauses(context,
                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  sizeof(cl_long) * (formula.totalNumVar),
                                  formula.clauses);
            cl::Buffer bufNumVarsC(context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(cl_long) * (formula.numclauses),
                                   formula.numVarsC);
            kernel = cl::Kernel(program, "solveLeaf");
            kernel.setArg(0, bufClauses);
            kernel.setArg(1, bufNumVarsC);
            kernel.setArg(2, formula.numclauses);
            kernel.setArg(3, bufSol);
            kernel.setArg(4, node.numv);
            kernel.setArg(5, bufVertices);
            size_t numKernels = pow(2, node.numv);
            queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(numKernels));
            queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_long) * (node.numSol * (node.numv + 1)),
                                    node.solution);

//            solveLeaf(formula.clauses, formula.numVarsC, formula.numclauses, node.solution, node.numv,
//                      node.vertices);

        } else if (node.nume == 1 && decomp.bags[node.edges[0] - 1].numv < node.numv) {
            //introduce node
            cl::Buffer bufClauses(context,
                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  sizeof(cl_long) * (formula.totalNumVar),
                                  formula.clauses);
            cl::Buffer bufNumVarsC(context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(cl_long) * (formula.numclauses),
                                   formula.numVarsC);
            cl::Buffer bufSolNext(context,
                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  sizeof(cl_long) * (decomp.bags[node.edges[0] - 1].numSol *
                                                     (decomp.bags[node.edges[0] - 1].numv + 1)),
                                  decomp.bags[node.edges[0] - 1].solution);
            kernel = cl::Kernel(program, "solveIntroduce");
            kernel.setArg(0, bufClauses);
            kernel.setArg(1, bufNumVarsC);
            kernel.setArg(2, formula.numclauses);
            kernel.setArg(3, bufSol);
            kernel.setArg(4, node.numv);
            kernel.setArg(5, bufSolNext);
            kernel.setArg(6, decomp.bags[node.edges[0] - 1].numv);
            kernel.setArg(7, decomp.bags[node.edges[0] - 1].numSol);
            kernel.setArg(8, bufVertices);
            size_t numKernels = pow(2, node.numv);
            queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(numKernels));
            queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_long) * (node.numSol * (node.numv + 1)),
                                    node.solution);

//            solveIntroduce(formula.clauses, formula.numVarsC, formula.numclauses, node.solution, node.numv,
//                           decomp.bags[node.edges[0] - 1].solution,
//                           decomp.bags[node.edges[0] - 1].numv, decomp.bags[node.edges[0] - 1].numSol,
//                           node.vertices);

        } else {
            return;
        }
    }
}