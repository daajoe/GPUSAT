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
#include <chrono>

std::vector<cl::Platform> platforms;
cl::Context context;
std::vector<cl::Device> devices;
cl::CommandQueue queue;
cl::Program program;
cl::Kernel kernel;

void solveForgIntroduce(satformulaType &formula, bagType &node, bagType &next);

int main(int argc, char *argv[]) {
    long time_total = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
    ).count();
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

    long time_parsing = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
    ).count();
    treedecType treeDecomp = parseTreeDecomp(treeD.str());
    satformulaType satFormula = parseSatFormula(sat.str());
    time_parsing = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
    ).count() - time_parsing;

    try {
        long time_kernel = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
        ).count();
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
        std::string kernelStr = readFile("./kernel/SAT.cl");
        cl::Program::Sources sources(1, std::make_pair(kernelStr.c_str(),
                                                       kernelStr.length()));
        program = cl::Program(context, sources);
        program.build(devices);
        time_kernel = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
        ).count() - time_kernel;

        long time_solving = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
        ).count();
        solveProblem(treeDecomp, satFormula, treeDecomp.bags[0]);
        time_solving = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
        ).count() - time_solving;
        //printSolutions(treeDecomp);
        long time_model = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
        ).count();
        cl_long solutions = 0;
        for (int i = 0; i < treeDecomp.bags[0].numSol; i++) {
            solutions += treeDecomp.bags[0].solution[i];
        }
        std::cout << "{\n    \"Model Count\": " << solutions;
        if (solutions > 0) {
            cl_long *solution = new cl_long[satFormula.numVar + 1]();
            genSolution(treeDecomp, solution, treeDecomp.bags[0]);
            std::cout << "\n    ,\"Model\": \"";
            int i;
            for (i = 1; i <= satFormula.numVar - 1; i++) {
                cl_long assignment = solution[i];
                std::cout << solution[i] << ", ";
            }
            cl_long assignment = solution[i];
            std::cout << solution[i] << "\"";
        } else if (solutions == 0) {
            std::cout << "\n    ,\"Model\": \"UNSATISFIABLE\"";
        }
        time_model = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
        ).count() - time_model;
        time_total = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
        ).count() - time_total;
        std::cout << "\n    ,\"Time\":{";
        std::cout << "\n        \"Solving\": " << ((float) time_solving) / 1000;
        std::cout << "\n        ,\"Parsing\": " << ((float) time_parsing) / 1000;
        std::cout << "\n        ,\"Build_Kernel\": " << ((float) time_kernel) / 1000;
        std::cout << "\n        ,\"Generate_Model\": " << ((float) time_model) / 1000;
        std::cout << "\n        ,\"Total\": " << ((float) time_total) / 1000;
        std::cout << "\n    }";
        std::cout << "\n}";

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
        cl_long edge = node.edges[i] - 1;
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
            edge.solution = new cl_long[node.numSol]();
            bagType &next = decomp.bags[node.edges[i] - 1];
            solveForgIntroduce(formula, edge, next);
            //join
            solveJoin(node, node, edge);
        }
    }
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
    kernel.setArg(2, formula.numclauses);
    kernel.setArg(3, bufSol);
    kernel.setArg(4, node.numVars);
    kernel.setArg(5, bufSolNext);
    kernel.setArg(6, edge.numVars);
    size_t numKernels = node.numSol;
    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(numKernels));
    queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_long) * (node.numSol),
                            node.solution);
}

void solveLeaf(satformulaType &formula, bagType &node) {
    kernel = cl::Kernel(program, "solveLeaf");
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
    kernel.setArg(2, formula.numclauses);
    kernel.setArg(3, bufSol);
    kernel.setArg(4, node.numVars);
    size_t numKernels = node.numSol;
    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(numKernels));
    queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_long) * (node.numSol),
                            node.solution);
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
    size_t numKernels = edge.numSol;
    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(numKernels));
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
    size_t numKernels = node.numSol;
    queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(numKernels));
    queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_long) * (node.numSol), node.solution);
}

void genSolution(treedecType decomp, cl_long *solution, bagType node) {
    cl_long id = -1;
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
        cl_long assignment = (id & (1 << node.numVars - b - 1)) > 0 ? node.variables[b] : -node.variables[b];
        solution[node.variables[b]] = assignment;
    }
    for (int a = 0; a < node.numEdges; a++) {
        genSolEdge(decomp, solution, node, id, a);
    }
}

void genSolEdge(treedecType decomp, cl_long *solution, bagType lastNode, cl_long lastId, int edge) {
    bagType nextNode = decomp.bags[lastNode.edges[edge] - 1];
    cl_long nextId = 0;
    do {
        std::vector<cl_long> intersect_vars(nextNode.numVars + lastNode.numVars);
        std::vector<cl_long>::iterator it;
        it = set_intersection(nextNode.variables, nextNode.variables + nextNode.numVars, lastNode.variables,
                              lastNode.variables + lastNode.numVars, intersect_vars.begin());
        intersect_vars.resize(it - intersect_vars.begin());

        std::vector<cl_long> new_vars(nextNode.numVars + lastNode.numVars);
        it = set_difference(nextNode.variables,
                            nextNode.variables + nextNode.numVars, lastNode.variables,
                            lastNode.variables + lastNode.numVars, new_vars.begin());
        new_vars.resize(it - new_vars.begin());

        cl_long positionCurrent[intersect_vars.size()], positionNext[intersect_vars.size()], new_vars_position[new_vars.size()];

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

        cl_long template_id = 0;
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

