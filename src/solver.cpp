#include <algorithm>
#include <cmath>
#include <iostream>
#include <d4_utils.h>
#include <gpusautils.h>
#include <solver.h>

#define debug

namespace gpusat {
    void Solver::solveProblem(treedecType &decomp, satformulaType &formula, bagType &node) {
        if (isSat > 0) {
            node.solution = new solType[node.numSol]();
            if (node.numEdges == 0) {
                solveLeaf(formula, node);
            } else if (node.numEdges == 1) {
                cl_long edge = node.edges[0] - 1;
                solveProblem(decomp, formula, decomp.bags[edge]);
                if (isSat == 1) {
                    bagType &next = decomp.bags[node.edges[0] - 1];
                    solveForgIntroduce(formula, node, next);
                    delete[] next.solution;
                    delete[] next.variables;
                }
            } else if (node.numEdges > 1) {
                cl_long edge = node.edges[0] - 1;
                bagType &edge1 = decomp.bags[edge];
                solveProblem(decomp, formula, edge1);
                if (isSat <= 0) {
                    return;
                }
                if (isSat == 1) {
                    bagType tmp, edge2_, edge1_;
                    std::vector<cl_long> v1(static_cast<unsigned long long int>(edge1.numVars + node.numVars));
                    auto it1 = std::set_intersection(node.variables, node.variables + node.numVars, edge1.variables, edge1.variables + edge1.numVars,
                                                     v1.begin());
                    v1.resize(static_cast<unsigned long long int>(it1 - v1.begin()));
                    edge1_.variables = &v1[0];
                    edge1_.numVars = v1.size();
                    edge1_.numSol = static_cast<cl_long>(pow(2, edge1_.numVars));
                    edge1_.solution = new solType[edge1_.numSol]();
                    solveForget(edge1_, edge1);
                    delete[] edge1.solution;
                    delete[] edge1.variables;

                    for (int i = 1; i < node.numEdges; i++) {
                        edge = node.edges[i] - 1;
                        bagType &edge2 = decomp.bags[edge];
                        solveProblem(decomp, formula, edge2);
                        if (isSat <= 0) {
                            return;
                        }
                        std::vector<cl_long> v2(static_cast<unsigned long long int>(edge2.numVars + node.numVars));
                        auto it2 = std::set_intersection(node.variables, node.variables + node.numVars, edge2.variables,
                                                         edge2.variables + edge2.numVars, v2.begin());
                        v2.resize(static_cast<unsigned long long int>(it2 - v2.begin()));
                        edge2_.variables = &v2[0];
                        edge2_.numVars = v2.size();
                        edge2_.numSol = static_cast<cl_long>(pow(2, edge2_.numVars));
                        edge2_.solution = new solType[edge2_.numSol]();
                        solveForget(edge2_, edge2);
                        delete[] edge2.solution;
                        delete[] edge2.variables;

                        std::vector<cl_long> vt(static_cast<unsigned long long int>(edge1_.numVars + edge2_.numVars));
                        auto itt = std::set_union(edge1_.variables, edge1_.variables + edge1_.numVars, edge2_.variables,
                                                  edge2_.variables + edge2_.numVars, vt.begin());
                        vt.resize(static_cast<unsigned long long int>(itt - vt.begin()));
                        tmp.variables = new cl_long[vt.size()];
                        memcpy(tmp.variables, &vt[0], vt.size() * sizeof(cl_long));
                        tmp.numVars = vt.size();
                        tmp.numSol = static_cast<cl_long>(pow(2, tmp.numVars));
                        tmp.solution = new solType[tmp.numSol]();
                        solveJoin(tmp, edge1_, edge2_);

                        delete[] edge1_.solution;
                        delete[] edge2_.solution;

                        if (i > 1) {
                            delete[] edge1_.variables;
                        }
                        edge1_ = tmp;

                        if (i == node.numEdges - 1) {
                            solveIntroduce(formula, node, tmp);
                            delete[] tmp.solution;
                            delete[] tmp.variables;
                        }
                    }
                }
            }
        }
    }

    void Solver::solveForgIntroduce(satformulaType &formula, bagType &node, bagType &next) {

        std::vector<cl_long> diff_forget(static_cast<unsigned long long int>(next.numVars + node.numVars));
        std::vector<cl_long>::iterator it, it2;
        it = set_difference(next.variables, next.variables + next.numVars, node.variables, node.variables + node.numVars, diff_forget.begin());
        diff_forget.resize(static_cast<unsigned long long int>(it - diff_forget.begin()));
        std::vector<cl_long> diff_introduce(static_cast<unsigned long long int>(next.numVars + node.numVars));
        it = set_difference(node.variables, node.variables + node.numVars, next.variables, next.variables + next.numVars, diff_introduce.begin());
        diff_introduce.resize(static_cast<unsigned long long int>(it - diff_introduce.begin()));

        if (diff_introduce.empty()) {
            solveForget(node, next);
        } else if (diff_forget.empty()) {
            solveIntroduce(formula, node, next);
        } else {
            std::vector<cl_long> vect(static_cast<unsigned long long int>(next.numVars + node.numVars));
            it2 = set_difference(next.variables, next.variables + next.numVars, &diff_forget[0], &diff_forget[0] + diff_forget.size(), vect.begin());
            vect.resize(static_cast<unsigned long long int>(it2 - vect.begin()));
            bagType edge;
            edge.numVars = vect.size();
            edge.variables = &vect[0];
            edge.numSol = static_cast<cl_long>(pow(2, vect.size()));
            edge.solution = new solType[edge.numSol]();

            solveForget(edge, next);
            solveIntroduce(formula, node, edge);
            delete[] edge.solution;
        }
    }

    void Solver::solveIntroduce(satformulaType &formula, bagType &node, bagType &edge) {
        cl_long bagsizeEdge = static_cast<cl_long>(pow(2, std::min(maxWidth, edge.numVars)));
        cl_long bagsizeNode = static_cast<cl_long>(pow(2, std::min(maxWidth, node.numVars)));
        cl_long numIterations = pow(2, std::max((cl_long) 0, node.numVars - maxWidth));
        isSat = 0;
        for (int a = 0; a < numIterations; a++) {
            cl_long startIdNode = a * bagsizeNode;
            cl::Buffer bufSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagsizeNode),
                              &node.solution[a * bagsizeNode]);
            cl_long numSubIterations = pow(2, std::max((cl_long) 0, edge.numVars - maxWidth));
            for (int b = 0; b < numSubIterations; b++) {
                cl_long startIdEdge = b * bagsizeEdge;
                kernel = cl::Kernel(program, "solveIntroduce");
                cl::Buffer bufVertices;
                if (node.numVars > 0) {
                    bufVertices = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (node.numVars), node.variables);
                    kernel.setArg(7, bufVertices);
                } else {
                    kernel.setArg(7, NULL);
                }
                cl::Buffer bufClauses;
                if (formula.totalNumVar > 0) {
                    bufClauses = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (formula.totalNumVar),
                                            formula.clauses);
                    kernel.setArg(0, bufClauses);
                } else {
                    kernel.setArg(0, NULL);
                }
                cl::Buffer bufNumVarsC;
                if (formula.numclauses > 0) {
                    bufNumVarsC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (formula.numclauses),
                                             formula.numVarsC);
                    kernel.setArg(1, bufNumVarsC);
                } else {
                    kernel.setArg(1, NULL);
                }
                cl::Buffer bufSolNext(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagsizeEdge),
                                      &edge.solution[b * bagsizeEdge]);
                cl::Buffer bufNextVars;
                if (edge.numVars > 0) {
                    bufNextVars = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (edge.numVars), edge.variables);
                    kernel.setArg(8, bufNextVars);
                } else {
                    kernel.setArg(8, NULL);
                }
                cl::Buffer bufSAT(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &isSat);
                kernel.setArg(2, formula.numclauses);
                kernel.setArg(3, bufSol);
                kernel.setArg(4, node.numVars);
                kernel.setArg(5, bufSolNext);
                kernel.setArg(6, edge.numVars);
                kernel.setArg(9, bufSAT);
                cl_long minId = b * bagsizeEdge;
                cl_long maxId = (b + 1) * bagsizeEdge;
                cl::Buffer bufMinId(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &minId);
                cl::Buffer bufMaxId(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &maxId);
                kernel.setArg(10, bufMinId);
                kernel.setArg(11, bufMaxId);
                cl::Buffer bufStartNode(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &startIdNode);
                kernel.setArg(12, bufStartNode);
                cl::Buffer bufStartEdge(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &startIdEdge);
                kernel.setArg(13, bufStartEdge);
                queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(startIdNode)),
                                           cl::NDRange(static_cast<size_t>(bagsizeNode)));
                //queue.finish();
                queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(solType) * (bagsizeNode), &node.solution[startIdNode]);
                queue.enqueueReadBuffer(bufSAT, CL_TRUE, 0, sizeof(cl_long), &isSat);
            }
        }
        //std::cout << "Introduce\n";
        //GPUSATUtils::printSol(node.numSol,node.numVars,node.variables,node.solution);
    }

    void Solver::solveLeaf(satformulaType &formula, bagType &node) {
        cl_long bagsizeNode = static_cast<cl_long>(pow(2, std::min(maxWidth, node.numVars)));
        cl_long numIterations = pow(2, std::max((cl_long) 0, node.numVars - maxWidth));
        isSat = 0;
        for (int a = 0; a < numIterations; a++) {
            cl_long startIdNode = a * bagsizeNode;
            cl_int err;
            kernel = cl::Kernel(program, "solveLeaf", &err);
            cl::Buffer bufSol(context, CL_MEM_READ_WRITE, sizeof(solType) * (bagsizeNode));
            cl::Buffer bufVertices;
            if (node.numVars > 0) {
                bufVertices = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (node.numVars), node.variables);
                kernel.setArg(5, bufVertices);
            } else {
                kernel.setArg(5, NULL);
            }
            cl::Buffer bufClauses;
            if (formula.totalNumVar > 0) {
                bufClauses = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (formula.totalNumVar), formula.clauses);
                kernel.setArg(0, bufClauses);
            } else {
                kernel.setArg(0, NULL);
            }
            cl::Buffer bufNumVarsC;
            if (formula.numclauses > 0) {
                bufNumVarsC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (formula.numclauses), formula.numVarsC);
                kernel.setArg(1, bufNumVarsC);
            } else {
                kernel.setArg(1, NULL);
            }
            cl::Buffer bufSAT(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &isSat);
            kernel.setArg(2, formula.numclauses);
            kernel.setArg(3, bufSol);
            kernel.setArg(4, node.numVars);
            kernel.setArg(6, bufSAT);
            cl::Buffer bufStart(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &startIdNode);
            kernel.setArg(7, bufStart);
            queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(startIdNode)),
                                       cl::NDRange(static_cast<size_t>(bagsizeNode)));
            //queue.finish();
            queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(solType) * (bagsizeNode), &node.solution[startIdNode]);
            queue.enqueueReadBuffer(bufSAT, CL_TRUE, 0, sizeof(cl_long), &isSat);
        }
        //std::cout << "Leaf\n";
        //GPUSATUtils::printSol(node.numSol,node.numVars,node.variables,node.solution);
    }

    void Solver::solveForget(bagType &node, bagType &edge) {
        cl_long bagsizeEdge = static_cast<cl_long>(pow(2, std::min(maxWidth, edge.numVars)));
        cl_long bagsizeNode = static_cast<cl_long>(pow(2, std::min(maxWidth, node.numVars)));
        cl_long numIterations = pow(2, std::max((cl_long) 0, node.numVars - maxWidth));
        for (int a = 0; a < numIterations; a++) {
            cl_long startIdNode = a * bagsizeNode;
            cl::Buffer bufSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagsizeNode),
                              &node.solution[a * bagsizeNode]);
            cl_long numSubIterations = pow(2, std::max((cl_long) 0, edge.numVars - maxWidth));
            for (int b = 0; b < numSubIterations; b++) {
                cl_long startIdEdge = b * bagsizeEdge;
                kernel = cl::Kernel(program, "solveForget");
                cl::Buffer bufNextSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagsizeEdge),
                                      &edge.solution[b * bagsizeEdge]);
                cl::Buffer bufSolVars;
                if (node.numVars > 0) {
                    bufSolVars = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * node.numVars, node.variables);
                    kernel.setArg(1, bufSolVars);
                } else {
                    kernel.setArg(1, NULL);
                }
                cl::Buffer bufNextVars;
                if (edge.numVars > 0) {
                    bufNextVars = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * edge.numVars, edge.variables);
                    kernel.setArg(4, bufNextVars);
                } else {
                    kernel.setArg(4, NULL);
                }
                kernel.setArg(0, bufSol);
                kernel.setArg(2, bufNextSol);
                kernel.setArg(3, edge.numVars);
                kernel.setArg(5, (cl_long) pow(2, edge.numVars - node.numVars));
                kernel.setArg(6, node.numVars);
                cl_long minId = b * bagsizeEdge;
                cl_long maxId = (b + 1) * bagsizeEdge;
                cl::Buffer bufMinId(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &minId);
                cl::Buffer bufMaxId(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &maxId);
                kernel.setArg(7, bufMinId);
                kernel.setArg(8, bufMaxId);
                cl::Buffer bufStartNode(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &startIdNode);
                kernel.setArg(9, bufStartNode);
                cl::Buffer bufStartEdge(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &startIdEdge);
                kernel.setArg(10, bufStartEdge);
                queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(startIdNode)),
                                           cl::NDRange(static_cast<size_t>(bagsizeNode)));
                //queue.finish();
                queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(solType) * (bagsizeNode), &node.solution[startIdNode]);
            }
        }
        //std::cout << "Forget\n";
        //GPUSATUtils::printSol(node.numSol,node.numVars,node.variables,node.solution);
    }

    void Solver::solveJoin(bagType &node, bagType &edge1, bagType &edge2) {
        cl_long bagsizeEdge1 = static_cast<cl_long>(pow(2, std::min(maxWidth, edge1.numVars)));
        cl_long bagsizeEdge2 = static_cast<cl_long>(pow(2, std::min(maxWidth, edge2.numVars)));
        cl_long bagsizeNode = static_cast<cl_long>(pow(2, std::min(maxWidth, node.numVars)));
        for (cl_long i = 0; i < node.numSol; i++) {
            node.solution[i] = 1.0;
        }
        cl_long numIterations = pow(2, std::max((cl_long) 0, node.numVars - maxWidth));
        for (int a = 0; a < numIterations; a++) {
            cl_long startIdNode = a * bagsizeNode;
            cl_long startIdEdge1;
            cl_long startIdEdge2;
            cl_long numSubIterations = pow(2, std::max((cl_long) 0, std::max(edge1.numVars - maxWidth, edge2.numVars - maxWidth)));
            for (int b = 0; b < numSubIterations; b++) {
                cl::Buffer bufSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagsizeNode), &node.solution[a * bagsizeNode]);
                startIdEdge1 = b * bagsizeEdge1;
                startIdEdge2 = b * bagsizeEdge2;
                kernel = cl::Kernel(program, "solveJoin");
                cl_long es1 = (b < pow(2, std::max((cl_long) 0, edge1.numVars - maxWidth))) || b == 0 ? bagsizeEdge1 : 0;
                cl::Buffer bufSol1;
                if (es1 > 0) {
                    bufSol1 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (es1),
                                         &edge1.solution[(b < pow(2, std::max((cl_long) 0, edge1.numVars - maxWidth)) ? b : 0) * bagsizeEdge1]);
                    kernel.setArg(1, bufSol1);
                } else {
                    kernel.setArg(1, NULL);
                }
                cl_long es2 = (b < pow(2, std::max((cl_long) 0, edge2.numVars - maxWidth))) || b == 0 ? bagsizeEdge2 : 0;
                cl::Buffer bufSol2;
                if (es2 > 0) {
                    bufSol2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (es2),
                                         &edge2.solution[(b < pow(2, std::max((cl_long) 0, edge2.numVars - maxWidth)) ? b : 0) * bagsizeEdge2]);
                    kernel.setArg(2, bufSol2);
                } else {

                    kernel.setArg(2, NULL);
                }
                cl::Buffer bufSolVars;
                if (node.numVars > 0) {
                    bufSolVars = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * node.numVars, node.variables);
                    kernel.setArg(3, bufSolVars);
                } else {
                    kernel.setArg(3, NULL);
                }
                cl::Buffer bufSolVars1;
                if (edge1.numVars > 0) {
                    bufSolVars1 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * edge1.numVars, edge1.variables);
                    kernel.setArg(4, bufSolVars1);
                } else {
                    kernel.setArg(4, NULL);
                }
                cl::Buffer bufSolVars2;
                if (edge2.numVars > 0) {
                    bufSolVars2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * edge2.numVars, edge2.variables);
                    kernel.setArg(5, bufSolVars2);
                } else {
                    kernel.setArg(5, NULL);
                }
                kernel.setArg(0, bufSol);
                kernel.setArg(6, node.numVars);
                kernel.setArg(7, edge1.numVars);
                kernel.setArg(8, edge2.numVars);
                cl_long minId1 = ((b + 1) <= pow(2, std::max((cl_long) 0, edge1.numVars - maxWidth)) || b == 0 ? bagsizeEdge1 * b : -1);
                cl_long maxId1 = ((b + 1) <= pow(2, std::max((cl_long) 0, edge1.numVars - maxWidth)) || b == 0 ? bagsizeEdge1 * (b + 1) : -1);
                cl::Buffer bufMinId1(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &minId1);
                cl::Buffer bufMaxId1(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &maxId1);
                kernel.setArg(9, bufMinId1);
                kernel.setArg(10, bufMaxId1);
                cl_long minId2 = ((b + 1) <= pow(2, std::max((cl_long) 0, edge2.numVars - maxWidth)) || b == 0 ? bagsizeEdge2 * b : -1);
                cl_long maxId2 = ((b + 1) <= pow(2, std::max((cl_long) 0, edge2.numVars - maxWidth)) || b == 0 ? bagsizeEdge2 * (b + 1) : -1);
                cl::Buffer bufMinId2(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &minId2);
                cl::Buffer bufMaxId2(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &maxId2);
                kernel.setArg(11, bufMinId2);
                kernel.setArg(12, bufMaxId2);
                cl::Buffer bufStartNode(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &startIdNode);
                kernel.setArg(13, bufStartNode);
                cl::Buffer bufStartEdge1(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &startIdEdge1);
                kernel.setArg(14, bufStartEdge1);
                cl::Buffer bufStartEdge2(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &startIdEdge2);
                kernel.setArg(15, bufStartEdge2);
                queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(startIdNode)),
                                           cl::NDRange(static_cast<size_t>(bagsizeNode)));
                //queue.finish();
                queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(solType) * (bagsizeNode), &node.solution[startIdNode]);
            }
        }
        //std::cout << "Join\n";
        //GPUSATUtils::printSol(node.numSol,node.numVars,node.variables,node.solution);
    }
}