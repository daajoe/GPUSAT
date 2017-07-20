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
#include <main.h>
#include <sys/stat.h>
#include <numeric>
#include <cstdlib>

std::vector<cl::Platform> platforms;
cl::Context context;
std::vector<cl::Device> devices;
cl::CommandQueue queue;
cl::Program program;
cl::Kernel kernel;
cl_long isSat = 1;

void solveForgIntroduce(satformulaType &formula, bagType &node, bagType &next);

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
    treedecType treeDecomp = parseTreeDecomp(treeD.str());
    satformulaType satFormula = parseSatFormula(sat.str());
    time_parsing = getTime() - time_parsing;

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
            std::string kernelStr = readFile(sourcePath.c_str());
            cl::Program::Sources sources(1, std::make_pair(kernelStr.c_str(),
                                                           kernelStr.length()));
            program = cl::Program(context, sources);
            program.build(devices);

            const std::vector<size_t> binSizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
            std::vector<char> binData((unsigned long long int) std::accumulate(binSizes.begin(), binSizes.end(), 0));
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
            std::string kernelStr = readBinary(binPath.c_str());
            cl::Program::Binaries bins(1, std::make_pair((const void *) kernelStr.data(), kernelStr.size()));
            program = cl::Program(context, devices, bins, NULL, &err);
            program.build(devices);
        }
        time_build_kernel = getTime() - time_build_kernel;

        long long int time_solving = getTime();
        solveProblem(treeDecomp, satFormula, treeDecomp.bags[0]);
        time_solving = getTime() - time_solving;

        long long int time_model = getTime();
        if (isSat > 0) {
            cl_long solutions = 0;
            for (cl_long i = 0; i < treeDecomp.bags[0].numSol; i++) {
                bagType &n = treeDecomp.bags[0];
                solutions += treeDecomp.bags[0].solution[i];
            }
            std::cout << "{\n    \"Model Count\": " << solutions;
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

void
solveProblem(treedecType& decomp, satformulaType& formula, bagType& node) {

    for (int i = 0; i < node.numEdges; i++) {
        cl_long edge = node.edges[i] - 1;
        solveProblem(decomp, formula, decomp.bags[edge]);
    }

    if (isSat > 0) {
        node.solution = new cl_long[node.numSol]();
        if (node.numEdges == 0) {
            //leaf node
            solveLeaf(formula, node);
        } else if (node.numEdges == 1) {
            bagType &next = decomp.bags[node.edges[0] - 1];
            solveForgIntroduce(formula, node, next);

        } else if (node.numEdges > 1) {
            bagType &next = decomp.bags[node.edges[0] - 1];
            solveForgIntroduce(formula, node, next);
            for (int i = 1; i < node.numEdges; i++) {
                bagType edge;
                edge.numEdges = node.numEdges;
                edge.numSol = node.numSol;
                edge.numVars = node.numVars;
                edge.edges = node.edges;
                edge.variables = node.variables;
                edge.solution = new cl_long[node.numSol]();
                bagType &next = decomp.bags[node.edges[i] - 1];
                solveForgIntroduce(formula, edge, next);
                //join
                solveJoin(node, node, edge);
            }
        }
    }
    /*for (int i = 0; i < node.numEdges; i++) {
        delete[] decomp.bags[node.edges[i]-1].edges;
        delete[] decomp.bags[node.edges[i]-1].solution;
        delete[] decomp.bags[node.edges[i]-1].variables;
    }*/
}

void solveForgIntroduce(satformulaType &formula, bagType &node, bagType &next) {
    std::vector<cl_long> diff_forget(next.numVars + node.numVars);
    std::vector<cl_long>::iterator it, it2;
    it = set_difference(next.variables,
                        next.variables + next.numVars,
                        node.variables, node.variables + node.numVars, diff_forget.begin());
    diff_forget.resize(it - diff_forget.begin());
    std::vector<cl_long> diff_introduce(next.numVars + node.numVars);
    it = set_difference(node.variables, node.variables + node.numVars,
                        next.variables,
                        next.variables + next.numVars,
                        diff_introduce.begin());
    diff_introduce.resize(it - diff_introduce.begin());

    if (diff_introduce.size() == 0) {
        solveForget(node, next);
    } else if (diff_forget.size() == 0) {
        solveIntroduce(formula, node, next);
    } else {
        std::vector<cl_long> vect(next.numVars + node.numVars);
        it2 = set_difference(next.variables,
                             next.variables + next.numVars,
                             &diff_forget[0], &diff_forget[0] + diff_forget.size(),
                             vect.begin());
        vect.resize(it2 - vect.begin());
        bagType edge;
        edge.numVars = vect.size();
        edge.variables = &vect[0];
        edge.numSol = pow(2, vect.size());
        edge.solution = new cl_long[edge.numSol]();

        solveForget(edge, next);
        solveIntroduce(formula, node, edge);
    }
}

void solveIntroduce(satformulaType &formula, bagType &node, bagType &edge) {
    kernel = cl::Kernel(program, "solveIntroduce");
    cl::Buffer bufSol(context,
                      CL_MEM_READ_WRITE,
                      sizeof(cl_long) * (node.numSol));
    cl::Buffer bufVertices;
    if (node.numVars > 0) {
        bufVertices = cl::Buffer(context,
                                 CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 sizeof(cl_long) * (node.numVars),
                                 node.variables);
        kernel.setArg(7, bufVertices);
    } else {
        kernel.setArg(7, NULL);
    }
    cl::Buffer bufClauses;
    if (formula.totalNumVar > 0) {
        bufClauses = cl::Buffer(context,
                                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(cl_long) * (formula.totalNumVar),
                                formula.clauses);
        kernel.setArg(0, bufClauses);
    } else {
        kernel.setArg(0, NULL);
    }
    cl::Buffer bufNumVarsC;
    if (formula.numclauses > 0) {
        bufNumVarsC = cl::Buffer(context,
                                 CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 sizeof(cl_long) * (formula.numclauses),
                                 formula.numVarsC);
        kernel.setArg(1, bufNumVarsC);
    } else {
        kernel.setArg(1, NULL);
    }
    cl::Buffer bufSolNext(context,
                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_long) * (edge.numSol),
                          edge.solution);
    cl::Buffer bufNextVars;
    if (edge.numVars > 0) {
        bufNextVars = cl::Buffer(context,
                                 CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 sizeof(cl_long) * (edge.numVars),
                                 edge.variables);
        kernel.setArg(8, bufNextVars);
    } else {
        kernel.setArg(8, NULL);
    }
    isSat = 0;
    cl::Buffer bufSAT(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &isSat);
    kernel.setArg(2, formula.numclauses);
    kernel.setArg(3, bufSol);
    kernel.setArg(4, node.numVars);
    kernel.setArg(5, bufSolNext);
    kernel.setArg(6, edge.numVars);
    kernel.setArg(9, bufSAT);
    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(node.numSol));
    queue.finish();
    queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_long) * (node.numSol), node.solution);
    queue.enqueueReadBuffer(bufSAT, CL_TRUE, 0, sizeof(cl_long), &isSat);
}

void solveLeaf(satformulaType &formula, bagType &node) {
    cl_int err;
    kernel = cl::Kernel(program, "solveLeaf", &err);
    cl::Buffer bufSol(context,
                      CL_MEM_READ_WRITE,
                      sizeof(cl_long) * (node.numSol));
    cl::Buffer bufVertices;
    if (node.numVars > 0) {
        bufVertices = cl::Buffer(context,
                                 CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 sizeof(cl_long) * (node.numVars),
                                 node.variables);
        kernel.setArg(5, bufVertices);
    } else {
        kernel.setArg(5, NULL);
    }
    cl::Buffer bufClauses;
    if (formula.totalNumVar > 0) {
        bufClauses = cl::Buffer(context,
                                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(cl_long) * (formula.totalNumVar),
                                formula.clauses);
        kernel.setArg(0, bufClauses);
    } else {
        kernel.setArg(0, NULL);
    }
    cl::Buffer bufNumVarsC;
    if (formula.numclauses > 0) {
        bufNumVarsC = cl::Buffer(context,
                                 CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 sizeof(cl_long) * (formula.numclauses),
                                 formula.numVarsC);
        kernel.setArg(1, bufNumVarsC);
    } else {
        kernel.setArg(1, NULL);
    }
    isSat = 0;
    cl::Buffer bufSAT(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &isSat);
    kernel.setArg(2, formula.numclauses);
    kernel.setArg(3, bufSol);
    kernel.setArg(4, node.numVars);
    kernel.setArg(6, bufSAT);
    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(node.numSol));
    queue.finish();
    queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_long) * (node.numSol), node.solution);
    queue.enqueueReadBuffer(bufSAT, CL_TRUE, 0, sizeof(cl_long), &isSat);
}

void solveForget(bagType &node, bagType &edge) {
    kernel = cl::Kernel(program, "solveForget");
    cl::Buffer bufSol(context,
                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                      sizeof(cl_long) * (node.numSol),
                      node.solution);
    cl::Buffer bufNextSol(context,
                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_long) * (edge.numSol),
                          edge.solution);
    cl::Buffer bufSolVars;
    if (node.numVars > 0) {
        bufSolVars = cl::Buffer(context,
                                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(cl_long) * node.numVars,
                                node.variables);
        kernel.setArg(1, bufSolVars);
    } else {
        kernel.setArg(1, NULL);
    }
    cl::Buffer bufNextVars;
    if (edge.numVars > 0) {
        bufNextVars = cl::Buffer(context,
                                 CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 sizeof(cl_long) * edge.numVars,
                                 edge.variables);
        kernel.setArg(4, bufNextVars);
    } else {
        kernel.setArg(4, NULL);
    }
    kernel.setArg(0, bufSol);
    kernel.setArg(2, bufNextSol);
    kernel.setArg(3, edge.numVars);
    kernel.setArg(5, (cl_long) pow(2, edge.numVars - node.numVars));
    kernel.setArg(6, node.numVars);
    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(edge.numSol));
    queue.finish();
    queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_long) * (node.numSol), node.solution);
}

void solveJoin(bagType &node, bagType &edge1, bagType &edge2) {
    kernel = cl::Kernel(program, "solveJoin");
    cl::Buffer bufSol(context,
                      CL_MEM_READ_WRITE,
                      sizeof(cl_long) * (node.numSol));
    cl_long *solutions = new cl_long[node.numSol * 2]();
    std::copy(edge1.solution, edge1.solution + node.numSol,
              solutions);
    std::copy(edge2.solution, edge2.solution + node.numSol,
              &solutions[node.numSol]);
    cl::Buffer bufSolOther(context,
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_long) * (node.numSol * 2),
                           solutions);
    kernel.setArg(0, bufSol);
    kernel.setArg(1, bufSolOther);
    kernel.setArg(2, node.numSol);
    size_t numKernels = (size_t) node.numSol;
    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(numKernels));
    queue.finish();
    queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_long) * (node.numSol), node.solution);
}
