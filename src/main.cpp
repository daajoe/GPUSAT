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

void solveForgIntroduce(satformulaType &formula, bagType &node, bagType &next);

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
        std::string kernelStr = readFile("./kernel/SAT.cl");
        cl::Program::Sources sources(1, std::make_pair(kernelStr.c_str(),
                                                       kernelStr.length()));
        program = cl::Program(context, sources);
        program.build(devices);
        solveProblem(treeDecomp, satFormula, treeDecomp.bags[0]);
        //printSolutions(treeDecomp);
        int solutions = 0;
        for (int i = 0; i < treeDecomp.bags[0].numSol; i++) {
            solutions += treeDecomp.bags[0].solution[i];
        }
        std::cout << "Model Count: " << solutions;
        if (solutions > 0) {
            cl_int *solution = new cl_int[satFormula.numVar + 1]();
            genSolution(treeDecomp, solution, treeDecomp.bags[0]);
            std::cout << "\nModel: { ";
            for (int i = 0; i <= satFormula.numVar; i++) {
                cl_int assignment = solution[i];
                if (solution[i] > 0)
                    std::cout << solution[i] << " ";
            }
            std::cout << "}";
        } else if (solutions == 0) {
            std::cout << "\nUnsat";
        }

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

    for (int i = 0; i < node.numEdges; i++) {
        cl_int edge = node.edges[i] - 1;
        solveProblem(decomp, formula, decomp.bags[edge]);
    }

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
            edge.solution = new cl_int[node.numSol];
            bagType &next = decomp.bags[node.edges[i] - 1];
            solveForgIntroduce(formula, edge, next);
            //join
            solveJoin(node, node, edge);
        }
    }
}

void solveForgIntroduce(satformulaType &formula, bagType &node, bagType &next) {
    std::vector<cl_int> diff_forget(next.numVars + node.numVars);
    std::vector<cl_int>::iterator it, it2;
    it = set_difference(next.variables,
                        next.variables + next.numVars,
                        node.variables, node.variables + node.numVars, diff_forget.begin());
    diff_forget.resize(it - diff_forget.begin());
    std::vector<cl_int> diff_introduce(next.numVars + node.numVars);
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
        std::vector<cl_int> vect(next.numVars + node.numVars);
        it2 = set_difference(next.variables,
                             next.variables + next.numVars,
                             &diff_forget[0], &diff_forget[0] + diff_forget.size(),
                             vect.begin());
        vect.resize(it2 - vect.begin());
        bagType edge;
        edge.numVars = vect.size();
        edge.variables = &vect[0];
        edge.numSol = pow(2, vect.size());
        edge.solution = new cl_int[edge.numSol];

        solveForget(edge, next);
        solveIntroduce(formula, node, edge);
    }
}

void solveIntroduce(satformulaType &formula, bagType &node, bagType &edge) {
    cl::Buffer bufSol(context,
                      CL_MEM_READ_WRITE,
                      sizeof(cl_int) * (node.numSol));
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
                          sizeof(cl_int) * (edge.numSol),
                          edge.solution);
    cl::Buffer bufNextVars(context,
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_int) * (edge.numVars),
                           edge.variables);
    kernel = cl::Kernel(program, "solveIntroduce");
    kernel.setArg(0, bufClauses);
    kernel.setArg(1, bufNumVarsC);
    kernel.setArg(2, formula.numclauses);
    kernel.setArg(3, bufSol);
    kernel.setArg(4, node.numVars);
    kernel.setArg(5, bufSolNext);
    kernel.setArg(6, edge.numVars);
    kernel.setArg(7, bufVertices);
    kernel.setArg(8, bufNextVars);
    size_t numKernels = node.numSol;
    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(numKernels));
    queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_int) * (node.numSol),
                            node.solution);
}

void solveLeaf(satformulaType &formula, bagType &node) {
    cl::Buffer bufSol(context,
                      CL_MEM_READ_WRITE,
                      sizeof(cl_int) * (node.numSol));
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
    queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_int) * (node.numSol),
                            node.solution);
}

void solveForget(bagType &node, bagType &edge) {
    cl::Buffer bufSol(context,
                      CL_MEM_READ_WRITE,
                      sizeof(cl_int) * (node.numSol));
    cl::Buffer bufVertices(context,
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_int) * (node.numVars),
                           node.variables);
    cl::Buffer bufNextSol(context,
                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_int) * (edge.numSol),
                          edge.solution);
    cl::Buffer bufSolVars(context,
                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_int) * node.numVars,
                          node.variables);
    cl::Buffer bufNextVars(context,
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_int) * edge.numVars,
                           edge.variables);
    kernel = cl::Kernel(program, "solveForget");
    kernel.setArg(0, bufSol);
    kernel.setArg(1, node.numVars);
    kernel.setArg(2, bufSolVars);
    kernel.setArg(3, bufNextSol);
    kernel.setArg(4, edge.numVars);
    kernel.setArg(5, bufNextVars);
    size_t numKernels = edge.numSol;
    queue.enqueueFillBuffer(bufSol, 0, 0, sizeof(cl_int) * (node.numSol));
    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(numKernels));
    queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_int) * (node.numSol), node.solution);
}

void solveJoin(bagType &node, bagType &edge1, bagType &edge2) {
    cl::Buffer bufSol(context,
                      CL_MEM_READ_WRITE,
                      sizeof(cl_int) * (node.numSol));
    cl::Buffer bufVertices(context,
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_int) * (node.numVars),
                           node.variables);
    cl_int *solutions = new cl_int[node.numSol * 2];
    std::copy(edge1.solution, edge1.solution + node.numSol,
              solutions);
    std::copy(edge2.solution, edge2.solution + node.numSol,
              &solutions[node.numSol]);
    cl::Buffer bufSolOther(context,
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(cl_int) * (node.numSol * 2),
                           solutions);
    kernel = cl::Kernel(program, "solveJoin");
    kernel.setArg(0, bufSol);
    kernel.setArg(1, bufSolOther);
    kernel.setArg(2, node.numSol);
    size_t numKernels = node.numSol;
    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(numKernels));
    queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_int) * (node.numSol), node.solution);
}

void genSolution(treedecType decomp, cl_int *solution, bagType node) {
    cl_int id = -1;
    for (int i = 0; i < node.numSol; i++) {
        if (node.solution[i] > 0) {
            id = i;
            break;
        }
    }
    //no solution found.
    if (id < 0) {
        return;
    }
    for (int b = 0; b < node.numVars; b++) {
        cl_int assignment = (id & (1 << node.numVars - b - 1)) > 0 ? node.variables[b] : -node.variables[b];
        solution[node.variables[b]] = assignment;
    }
    for (int a = 0; a < node.numEdges; a++) {
        genSolEdge(decomp, solution, node, id, a);
    }
}

void genSolEdge(treedecType decomp, cl_int *solution, bagType lastNode, cl_int lastId, int edge) {
    bagType nextNode = decomp.bags[lastNode.edges[edge] - 1];
    cl_int nextId = 0;
    do {
        std::vector<cl_int> intersect_vars(nextNode.numVars + lastNode.numVars);
        std::vector<cl_int>::iterator it;
        it = set_intersection(nextNode.variables, nextNode.variables + nextNode.numVars, lastNode.variables,
                              lastNode.variables + lastNode.numVars, intersect_vars.begin());
        intersect_vars.resize(it - intersect_vars.begin());

        std::vector<cl_int> new_vars(nextNode.numVars + lastNode.numVars);
        it = set_difference(nextNode.variables,
                            nextNode.variables + nextNode.numVars, lastNode.variables,
                            lastNode.variables + lastNode.numVars, new_vars.begin());
        new_vars.resize(it - new_vars.begin());

        cl_int positionCurrent[intersect_vars.size()], positionNext[intersect_vars.size()], new_vars_position[new_vars.size()];

        for (int a = 0, b = 0; a < lastNode.numVars; a++) {
            if (lastNode.variables[a] == intersect_vars[b]) {
                positionCurrent[b] = lastNode.numVars - a - 1;
                b++;
            }
        }
        for (int a = 0, b = 0, c = 0; a < nextNode.numVars; a++) {
            if (nextNode.variables[a] == intersect_vars[b]) {
                positionNext[b] = nextNode.numVars - a - 1;
                b++;
            }
            if (nextNode.variables[a] == new_vars[c]) {
                new_vars_position[c] = nextNode.numVars - a - 1;
                c++;
            }
        }

        cl_int template_id = 0;
        for (int a = 0; a < intersect_vars.size(); a++) {
            template_id = template_id | ((lastId & (1 << positionCurrent[a])) >> positionCurrent[a] << positionNext[a]);
        }

        nextId = template_id;
        if (new_vars.size() == 0) {
            for (int b = 0; b < nextNode.numVars; b++) {
                solution[nextNode.variables[b]] =
                        (nextId & (1 << nextNode.numVars - b - 1)) > 0 ? nextNode.variables[b]
                                                                       : -nextNode.variables[b];
            }
        } else {
            for (int a = 0; a < pow(2, new_vars.size()); a++) {
                nextId = template_id;
                for (int b = 0; b < new_vars.size(); b++) {
                    nextId = nextId | ((a & (1 << (new_vars.size() - b - 1))) >> (new_vars.size() - b - 1)
                                                                              << new_vars_position[b]);
                }
                if (nextNode.solution[nextId] > 0) {
                    for (int b = 0; b < nextNode.numVars; b++) {
                        solution[nextNode.variables[b]] =
                                (nextId & (1 << nextNode.numVars - b - 1)) > 0 ? nextNode.variables[b]
                                                                               : -nextNode.variables[b];
                    }
                    break;
                }
            }
        }

        lastNode = nextNode;
        lastId = nextId;
        if (lastNode.numEdges > 1) {
            for (int i = 1; i < lastNode.numEdges; i++) {
                genSolEdge(decomp, solution, lastNode, lastId, i);
            }
        }
        if (lastNode.numEdges > 0) {
            nextNode = decomp.bags[lastNode.edges[0] - 1];
        }
    } while (lastNode.numEdges != 0);
}

