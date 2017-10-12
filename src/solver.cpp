#include <algorithm>
#include <cmath>
#include <iostream>
#include <d4_utils.h>
#include <gpusautils.h>
#include <solver.h>

namespace gpusat {
    void Solver::solveProblem(treedecType &decomp, satformulaType &formula, bagType &node) {
        if (isSat > 0) {
            if (node.numEdges == 0) {
                solveLeaf(formula, node);
            } else if (node.numEdges == 1) {
                cl_long edge = node.edges[0] - 1;
                solveProblem(decomp, formula, decomp.bags[edge]);
                if (isSat == 1) {
                    bagType &next = decomp.bags[node.edges[0] - 1];
                    solveForgIntroduce(formula, node, next);
                    delete next.solution;
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
                    solveForget(edge1_, edge1, formula);
                    delete edge1.solution;
                    delete[] edge1.variables;

                    for (int i = 1; i < node.numEdges; i++) {
                        edge = node.edges[i] - 1;
                        bagType &edge2 = decomp.bags[edge];
                        solveProblem(decomp, formula, edge2);
                        if (isSat <= 0) {
                            return;
                        }
                        std::vector<cl_long> v2(static_cast<unsigned long long int>(edge2.numVars + node.numVars));
                        auto it2 = std::set_intersection(node.variables, node.variables + node.numVars, edge2.variables, edge2.variables + edge2.numVars,
                                                         v2.begin());
                        v2.resize(static_cast<unsigned long long int>(it2 - v2.begin()));
                        edge2_.variables = &v2[0];
                        edge2_.numVars = v2.size();
                        edge2_.numSol = static_cast<cl_long>(pow(2, edge2_.numVars));
                        solveForget(edge2_, edge2, formula);
                        delete edge2.solution;
                        delete[] edge2.variables;

                        std::vector<cl_long> vt(static_cast<unsigned long long int>(edge1_.numVars + edge2_.numVars));
                        auto itt = std::set_union(edge1_.variables, edge1_.variables + edge1_.numVars, edge2_.variables, edge2_.variables + edge2_.numVars,
                                                  vt.begin());
                        vt.resize(static_cast<unsigned long long int>(itt - vt.begin()));
                        tmp.variables = new cl_long[vt.size()];
                        memcpy(tmp.variables, &vt[0], vt.size() * sizeof(cl_long));
                        tmp.numVars = vt.size();
                        tmp.numSol = static_cast<cl_long>(pow(2, tmp.numVars));
                        solveJoin(tmp, edge1_, edge2_, formula);

                        delete edge1_.solution;
                        delete edge2_.solution;

                        if (i > 1) {
                            delete[] edge1_.variables;
                        }
                        edge1_ = tmp;

                        if (i == node.numEdges - 1) {
                            if (inci) {
                                if (node.numVars != tmp.numVars) {
                                    solveIntroduce(formula, node, tmp);
                                    delete tmp.solution;
                                    delete[] tmp.variables;
                                } else {
                                    delete node.solution;
                                    delete[] node.variables;
                                    node.solution = tmp.solution;
                                    node.variables = tmp.variables;
                                }
                            } else {
                                solveIntroduce(formula, node, tmp);
                                delete tmp.solution;
                                delete[] tmp.variables;
                            }
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
            solveForget(node, next, formula);
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

            solveForget(edge, next, formula);
            solveIntroduce(formula, node, edge);
            delete edge.solution;
        }
    }

    void Solver_Primal::solveIntroduce(satformulaType &formula, bagType &node, bagType &edge) {
#ifdef DEBUG
        std::cout << "Solver_Primal::solveIntroduce:\n";
        std::cout.flush();
#endif
        node.solution = new std::vector<std::pair<cl_long, solType>>();
        node.numSol = 0;
        cl_long bagSizeEdge = static_cast<cl_long>(pow(2, std::min(maxWidth, edge.numVars)));
        cl_long bagSizeNode = static_cast<cl_long>(pow(2, std::min(maxWidth, node.numVars)));
        cl_long numIterations = static_cast<cl_long>(pow(2, std::max((cl_long) 0, node.numVars - maxWidth)));
        solType *eSols = new solType[bagSizeNode]();
        isSat = 0;
        for (int a = 0; a < numIterations; a++) {
            cl_long startIdNode = a * bagSizeNode;
            solType *sols = new solType[bagSizeNode]();
            cl::Buffer bufSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeNode), sols);
            cl_long numSubIterations = static_cast<cl_long>(pow(2, std::max((cl_long) 0, edge.numVars - maxWidth)));
            cl_long d = 0;
            for (int b = 0; b < numSubIterations; b++) {
                cl_long startIdEdge = b * bagSizeEdge;
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
                for (int i = 0; i < bagSizeEdge; i++) {
                    if (edge.solution->size() > d && edge.solution->at(d).first == b * bagSizeEdge + i) {
                        eSols[i] = edge.solution->at(d).second;
                        d++;
                    } else {
                        eSols[i] = 0.0;
                    }
                }
                cl::Buffer bufSolNext(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeEdge), eSols);
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
                cl_long minId = b * bagSizeEdge;
                cl_long maxId = (b + 1) * bagSizeEdge;
                cl::Buffer bufMinId(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &minId);
                cl::Buffer bufMaxId(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &maxId);
                kernel.setArg(10, bufMinId);
                kernel.setArg(11, bufMaxId);
                cl::Buffer bufStartNode(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &startIdNode);
                kernel.setArg(12, bufStartNode);
                cl::Buffer bufStartEdge(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &startIdEdge);
                kernel.setArg(13, bufStartEdge);
                cl::Buffer bufWeights;
                if (formula.variableWeights != nullptr) {
                    bufWeights = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * formula.numWeights, formula.variableWeights);
                    kernel.setArg(14, bufWeights);
                } else {
                    kernel.setArg(14, NULL);
                }

                queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(startIdNode)), cl::NDRange(static_cast<size_t>(bagSizeNode)));
                queue.finish();
                queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(solType) * (bagSizeNode), sols);
                queue.enqueueReadBuffer(bufSAT, CL_TRUE, 0, sizeof(cl_long), &isSat);
            }
            for (cl_long i = 0; i < bagSizeNode; i++) {
                if (sols[i] != 0) {
                    node.solution->push_back(std::pair<cl_long, solType>(i + a * bagSizeNode, sols[i]));
                    node.numSol++;
                }
            }
            delete[] sols;
        }
#ifdef DEBUG
        GPUSATUtils::printSol(node.numSol, node.numVars, node.variables, node.solution, formula);
#endif
    }

    void Solver_Primal::solveLeaf(satformulaType &formula, bagType &node) {
#ifdef DEBUG
        std::cout << "Solver_Primal::solveLeaf:\n";
        std::cout.flush();
#endif
        cl_long bagSizeNode = static_cast<cl_long>(pow(2, std::min(maxWidth, node.numVars)));
        cl_long numIterations = static_cast<cl_long>(pow(2, std::max((cl_long) 0, node.numVars - maxWidth)));
        node.solution = new std::vector<std::pair<cl_long, solType>>();
        node.numSol = 0;
        isSat = 0;
        for (int a = 0; a < numIterations; a++) {
            cl_long startIdNode = a * bagSizeNode;
            cl_int err;
            kernel = cl::Kernel(program, "solveLeaf", &err);
            cl::Buffer bufSol(context, CL_MEM_READ_WRITE, sizeof(solType) * (bagSizeNode));
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
            cl::Buffer bufWeights;
            if (formula.variableWeights != nullptr) {
                bufWeights = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * formula.numWeights, formula.variableWeights);
                kernel.setArg(8, bufWeights);
            } else {
                kernel.setArg(8, NULL);
            }

            queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(startIdNode)), cl::NDRange(static_cast<size_t>(bagSizeNode)));
            queue.finish();
            solType *sols = new solType[bagSizeNode];
            queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(solType) * (bagSizeNode), sols);
            queue.enqueueReadBuffer(bufSAT, CL_TRUE, 0, sizeof(cl_long), &isSat);
            for (cl_long i = 0; i < bagSizeNode; i++) {
                if (sols[i] != 0) {
                    node.solution->push_back(std::pair<cl_long, solType>(i + a * bagSizeNode, sols[i]));
                    node.numSol++;
                }
            }
            delete[] sols;
        }
#ifdef DEBUG
        GPUSATUtils::printSol(node.numSol, node.numVars, node.variables, node.solution, formula);
#endif
    }

    void Solver_Primal::solveForget(bagType &node, bagType &edge, satformulaType &formula) {
#ifdef DEBUG
        std::cout << "Solver_Primal::solveForget:\n";
        std::cout.flush();
#endif
        node.solution = new std::vector<std::pair<cl_long, solType>>();
        node.numSol = 0;
        cl_long bagSizeEdge = static_cast<cl_long>(pow(2, std::min(maxWidth, edge.numVars)));
        cl_long bagSizeNode = static_cast<cl_long>(pow(2, std::min(maxWidth, node.numVars)));
        cl_long numIterations = static_cast<cl_long>(pow(2, std::max((cl_long) 0, node.numVars - maxWidth)));
        solType *eSols = new solType[bagSizeEdge]();
        for (int a = 0; a < numIterations; a++) {
            cl_long startIdNode = a * bagSizeNode;
            solType *sols = new solType[bagSizeNode]();
            cl::Buffer bufSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeNode), sols);
            cl_long d = 0;
            cl_long numSubIterations = static_cast<cl_long>(pow(2, std::max((cl_long) 0, edge.numVars - maxWidth)));
            for (int b = 0; b < numSubIterations; b++) {
                cl_long startIdEdge = b * bagSizeEdge;
                kernel = cl::Kernel(program, "solveForget");
                for (int i = 0; i < bagSizeEdge; i++) {
                    if (edge.solution->size() > d && edge.solution->at(d).first == b * bagSizeEdge + i) {
                        eSols[i] = edge.solution->at(d).second;
                        d++;
                    } else {
                        eSols[i] = 0.0;
                    }
                }
                cl::Buffer bufNextSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeEdge), eSols);
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
                cl_long minId = b * bagSizeEdge;
                cl_long maxId = (b + 1) * bagSizeEdge;
                cl::Buffer bufMinId(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &minId);
                cl::Buffer bufMaxId(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &maxId);
                kernel.setArg(7, bufMinId);
                kernel.setArg(8, bufMaxId);
                cl::Buffer bufStartNode(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &startIdNode);
                kernel.setArg(9, bufStartNode);
                cl::Buffer bufStartEdge(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &startIdEdge);
                kernel.setArg(10, bufStartEdge);

                queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(startIdNode)), cl::NDRange(static_cast<size_t>(bagSizeNode)));
                queue.finish();
                queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(solType) * (bagSizeNode), sols);
            }
            for (cl_long i = 0; i < bagSizeNode; i++) {
                if (sols[i] != 0) {
                    node.solution->push_back(std::pair<cl_long, solType>(i + a * bagSizeNode, sols[i]));
                    node.numSol++;
                }
            }
            delete[] sols;
        }
#ifdef DEBUG
        GPUSATUtils::printSol(node.numSol, node.numVars, node.variables, node.solution, formula);
#endif
    }

    void Solver_Primal::solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula) {
#ifdef DEBUG
        std::cout << "Solver_Primal::solveJoin:\n";
        std::cout.flush();
#endif
        node.solution = new std::vector<std::pair<cl_long, solType>>();
        node.numSol = 0;
        cl_long bagSizeEdge1 = static_cast<cl_long>(pow(2, std::min(maxWidth, edge1.numVars)));
        cl_long bagSizeEdge2 = static_cast<cl_long>(pow(2, std::min(maxWidth, edge2.numVars)));
        cl_long bagSizeNode = static_cast<cl_long>(pow(2, std::min(maxWidth, node.numVars)));
        solType *eSols1 = new solType[bagSizeNode]();
        solType *eSols2 = new solType[bagSizeNode]();
        cl_long numIterations = static_cast<cl_long>(pow(2, std::max((cl_long) 0, node.numVars - maxWidth)));
        for (int a = 0; a < numIterations; a++) {
            cl_long startIdNode = a * bagSizeNode;
            cl_long startIdEdge1;
            cl_long startIdEdge2;
            cl_long numSubIterations = static_cast<cl_long>(pow(2, std::max((cl_long) 0, std::max(edge1.numVars - maxWidth, edge2.numVars - maxWidth))));
            cl_ulong d = 0, e = 0;
            solType *sols = new solType[bagSizeNode]();
            for (cl_long i = 0; i < bagSizeNode; i++) {
                sols[i] = 1.0;
            }
            for (int b = 0; b < numSubIterations; b++) {
                cl::Buffer bufSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeNode), sols);
                startIdEdge1 = b * bagSizeEdge1;
                startIdEdge2 = b * bagSizeEdge2;
                kernel = cl::Kernel(program, "solveJoin");
                cl_long es1 = (b < pow(2, std::max((cl_long) 0, edge1.numVars - maxWidth))) || b == 0 ? bagSizeEdge1 : 0;
                cl::Buffer bufSol1;
                if (es1 > 0) {
                    for (int i = 0; i < es1; i++) {
                        if (edge1.solution->size() > d && edge1.solution->at(d).first == bagSizeEdge1 * b + i) {
                            eSols1[i] = edge1.solution->at(d).second;
                            d++;
                        } else {
                            eSols1[i] = 0.0;
                        }
                    }
                    bufSol1 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (es1), eSols1);
                    kernel.setArg(1, bufSol1);
                } else {
                    kernel.setArg(1, NULL);
                }
                cl_long es2 = (b < pow(2, std::max((cl_long) 0, edge2.numVars - maxWidth))) || b == 0 ? bagSizeEdge2 : 0;
                cl::Buffer bufSol2;
                if (es2 > 0) {
                    for (int i = 0; i < es2; i++) {
                        if (edge2.solution->size() > e && edge2.solution->at(e).first == bagSizeEdge2 * b + i) {
                            solType s = edge2.solution->at(e).second;
                            eSols2[i] = edge2.solution->at(e).second;
                            e++;
                        } else {
                            eSols2[i] = 0.0;
                        }
                    }
                    bufSol2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (es2), eSols2);
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
                cl_long minId1 = ((b + 1) <= pow(2, std::max((cl_long) 0, edge1.numVars - maxWidth)) || b == 0 ? bagSizeEdge1 * b : -1);
                cl_long maxId1 = ((b + 1) <= pow(2, std::max((cl_long) 0, edge1.numVars - maxWidth)) || b == 0 ? bagSizeEdge1 * (b + 1) : -1);
                cl::Buffer bufMinId1(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &minId1);
                cl::Buffer bufMaxId1(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &maxId1);
                kernel.setArg(9, bufMinId1);
                kernel.setArg(10, bufMaxId1);
                cl_long minId2 = ((b + 1) <= pow(2, std::max((cl_long) 0, edge2.numVars - maxWidth)) || b == 0 ? bagSizeEdge2 * b : -1);
                cl_long maxId2 = ((b + 1) <= pow(2, std::max((cl_long) 0, edge2.numVars - maxWidth)) || b == 0 ? bagSizeEdge2 * (b + 1) : -1);
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
                cl::Buffer bufWeights;
                if (formula.variableWeights != nullptr) {
                    bufWeights = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * formula.numWeights, formula.variableWeights);
                    kernel.setArg(16, bufWeights);
                } else {
                    kernel.setArg(16, NULL);
                }

                queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(startIdNode)), cl::NDRange(static_cast<size_t>(bagSizeNode)));
                queue.finish();
                queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(solType) * (bagSizeNode), sols);
            }
            for (cl_long i = 0; i < bagSizeNode; i++) {
                if (sols[i] != 0) {
                    node.solution->push_back(std::pair<cl_long, solType>(i + a * bagSizeNode, sols[i]));
                    node.numSol++;
                }
            }
            delete[] sols;
        }
        delete[]eSols1;
        delete[]eSols2;
#ifdef DEBUG
        GPUSATUtils::printSol(node.numSol, node.numVars, node.variables, node.solution, formula);
#endif
    }

    void Solver_Incidence::solveIntroduce(satformulaType &formula, bagType &node, bagType &edge) {
#ifdef DEBUG
        std::cout << "Solver_Incidence::solveIntroduce:\n";
        std::cout.flush();
#endif
        node.solution = new std::vector<std::pair<cl_long, solType>>();
        node.numSol = 0;
        cl_long bagSizeEdge = static_cast<cl_long>(pow(2, std::min(maxWidth, edge.numVars)));
        cl_long bagSizeNode = static_cast<cl_long>(pow(2, std::min(maxWidth, node.numVars)));
        cl_long numIterations = static_cast<cl_long>(pow(2, std::max((cl_long) 0, node.numVars - maxWidth)));
        isSat = 0;

        std::vector<cl_long> clauseVector;
        for (int i = 0, a = 0, b = 0; i < node.numVars; i++) {
            if (node.variables[i] > formula.numVars) {
                for (; a < node.variables[i] - formula.numVars - 1; a++) {
                    b += formula.numVarsC[a];
                }
                for (int c = 0; c < formula.numVarsC[a]; c++) {
                    cl_long clause = formula.clauses[b + c];
                    clauseVector.push_back(clause);
                }
                clauseVector.push_back(0);
            }
        }

        std::vector<cl_long> nVars;
        std::vector<cl_long> nClauses;
        for (int i = 0; i < node.numVars; i++) {
            if (node.variables[i] > formula.numVars) {
                nClauses.push_back(node.variables[i]);
            } else {
                nVars.push_back(node.variables[i]);
            }
        }
        nVars.push_back(0);
        nClauses.push_back(0);

        std::vector<cl_long> eVars;
        std::vector<cl_long> eClauses;
        for (int i = 0; i < edge.numVars; i++) {
            if (edge.variables[i] > formula.numVars) {
                eClauses.push_back(edge.variables[i]);
            } else {
                eVars.push_back(edge.variables[i]);
            }
        }
        eVars.push_back(0);
        eClauses.push_back(0);
        solType *eSols = new solType[bagSizeNode]();

        for (int a = 0; a < numIterations; a++) {
            cl_long numSubIterations = static_cast<cl_long>(pow(2, std::max((cl_long) 0, edge.numVars - maxWidth)));
            solType *sols = new solType[bagSizeNode]();
            cl_long d = 0;
            for (int b = 0; b < numSubIterations; b++) {
                kernel = cl::Kernel(program, "solveIntroduce");
                //node solutions - 0
                cl::Buffer bufNSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeNode), sols);
                kernel.setArg(0, bufNSol);
                //edge solutions - 1
                for (int i = 0; i < bagSizeEdge; i++) {
                    if (edge.solution->size() > d && edge.solution->at(d).first == b * bagSizeEdge + i) {
                        eSols[i] = edge.solution->at(d).second;
                        d++;
                    } else {
                        eSols[i] = 0.0;
                    }
                }
                cl::Buffer bufESol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeEdge), eSols);
                kernel.setArg(1, bufESol);
                //clauses - 2
                cl::Buffer bufClauses = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (clauseVector.size()),
                                                   &clauseVector[0]);
                kernel.setArg(2, bufClauses);
                //clauses array length - 3
                cl_long clausesLength = clauseVector.size();
                cl::Buffer bufClausesLength(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &clausesLength);
                kernel.setArg(3, bufClausesLength);
                //node variables - 4
                cl::Buffer bufNVars = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * nVars.size(), &nVars[0]);
                kernel.setArg(4, bufNVars);
                //edge variables - 5
                cl::Buffer bufEVars = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * eVars.size(), &eVars[0]);
                kernel.setArg(5, bufEVars);
                //number of variables node - 6
                cl_long numNVars = nVars.size() - 1;
                cl::Buffer bufNumNVars(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &numNVars);
                kernel.setArg(6, bufNumNVars);
                //number of variables edge - 7
                cl_long numEVars = eVars.size() - 1;
                cl::Buffer bufNumEVars(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &numEVars);
                kernel.setArg(7, bufNumEVars);
                //node clauses - 8
                cl::Buffer bufNClauses(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * nClauses.size(), &nClauses[0]);
                kernel.setArg(8, bufNClauses);
                //edge clauses - 9
                cl::Buffer bufEClauses(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * eClauses.size(), &eClauses[0]);
                kernel.setArg(9, bufEClauses);
                //number of clauses node - 10
                cl_long numNClauses = nClauses.size() - 1;
                cl::Buffer bufNumNClauses(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &numNClauses);
                kernel.setArg(10, bufNumNClauses);
                //number of clauses edge - 11
                cl_long numEClauses = eClauses.size() - 1;
                cl::Buffer bufNumEClauses(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &numEClauses);
                kernel.setArg(11, bufNumEClauses);
                //start id node - 12
                cl_long startIdNode = a * bagSizeNode;
                cl::Buffer bufStartIdNode(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &startIdNode);
                kernel.setArg(12, bufStartIdNode);
                //start id edge - 13
                cl_long startIdEdge = b * bagSizeEdge;
                cl::Buffer bufStartIdEdge(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &startIdEdge);
                kernel.setArg(13, bufStartIdEdge);
                //min id edge - 14
                cl_long minId = b * bagSizeEdge;
                cl::Buffer bufMinId(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &minId);
                kernel.setArg(14, bufMinId);
                //max id edge - 15
                cl_long maxId = (b + 1) * bagSizeEdge;
                cl::Buffer bufMaxId(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &maxId);
                kernel.setArg(15, bufMaxId);
                //isSAT - 16
                cl::Buffer bufSAT(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &isSat);
                kernel.setArg(16, bufSAT);
                //weights - 17
                cl::Buffer bufWeights;
                if (formula.variableWeights != nullptr) {
                    bufWeights = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * formula.numWeights, formula.variableWeights);
                    kernel.setArg(17, bufWeights);
                } else {
                    kernel.setArg(17, NULL);
                }

                queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(startIdNode)), cl::NDRange(static_cast<size_t>(bagSizeNode)));
                queue.finish();
                queue.enqueueReadBuffer(bufNSol, CL_TRUE, 0, sizeof(solType) * (bagSizeNode), sols);
                queue.enqueueReadBuffer(bufSAT, CL_TRUE, 0, sizeof(cl_long), &isSat);
            }
            for (cl_long i = 0; i < bagSizeNode; i++) {
                if (sols[i] != 0) {
                    node.solution->push_back(std::pair<cl_long, solType>(i + a * bagSizeNode, sols[i]));
                    node.numSol++;
                }
            }
            delete[] sols;
        }
#ifdef DEBUG
        GPUSATUtils::printSol(node.numSol, node.numVars, node.variables, node.solution, formula);
#endif
    }

    void Solver_Incidence::solveLeaf(satformulaType &formula, bagType &node) {
#ifdef DEBUG
        std::cout << "Solver_Incidence::solveLeaf:\n";
        std::cout.flush();
#endif
        auto bagSizeNode = static_cast<cl_long>(pow(2, std::min(maxWidth, node.numVars)));
        cl_long numIterations = static_cast<cl_long>(pow(2, std::max((cl_long) 0, node.numVars - maxWidth)));
        node.solution = new std::vector<std::pair<cl_long, solType>>();
        node.numSol = 0;
        isSat = 0;

        std::vector<cl_long> clauseVector;
        for (int i = 0, a = 0, b = 0; i < node.numVars; i++) {
            if (node.variables[i] > formula.numVars) {
                for (; a < node.variables[i] - formula.numVars - 1; a++) {
                    b += formula.numVarsC[a];
                }
                for (int c = 0; c < formula.numVarsC[a]; c++) {
                    clauseVector.push_back(formula.clauses[b + c]);
                }
                clauseVector.push_back(0);
            }
        }

        std::vector<cl_long> nVars;
        std::vector<cl_long> nClauses;
        for (int i = 0; i < node.numVars; i++) {
            if (node.variables[i] > formula.numVars) {
                nClauses.push_back(node.variables[i]);
            } else {
                nVars.push_back(node.variables[i]);
            }
        }
        nVars.push_back(0);
        nClauses.push_back(0);

        for (int a = 0; a < numIterations; a++) {
            kernel = cl::Kernel(program, "solveLeaf");
            //node solutions - 0
            cl::Buffer bufSol(context, CL_MEM_READ_WRITE, sizeof(solType) * (bagSizeNode));
            kernel.setArg(0, bufSol);
            //clauses - 1
            cl::Buffer bufClauses = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (clauseVector.size()), &clauseVector[0]);
            kernel.setArg(1, bufClauses);
            //node variables - 2
            cl::Buffer bufVariables = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (nVars.size()), &nVars[0]);
            kernel.setArg(2, bufVariables);
            //number of clauses node - 3
            cl_long numNClauses = nClauses.size() - 1;
            cl::Buffer bufNumNClauses(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &numNClauses);
            kernel.setArg(3, bufNumNClauses);
            //start id node - 4
            cl_long startIdNode = a * bagSizeNode;
            cl::Buffer bufStartIDNode(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &startIdNode);
            kernel.setArg(4, bufStartIDNode);
            //models field - 5
            cl::Buffer bufSAT(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &isSat);
            kernel.setArg(5, bufSAT);
            //weights - 6
            cl::Buffer bufWeights;
            if (formula.variableWeights != nullptr) {
                bufWeights = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * formula.numWeights, formula.variableWeights);
                kernel.setArg(6, bufWeights);
            } else {
                kernel.setArg(6, NULL);
            }

            queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(startIdNode)), cl::NDRange(static_cast<size_t>(bagSizeNode)));
            queue.finish();
            solType *sols = new solType[bagSizeNode];
            queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(solType) * (bagSizeNode), sols);
            queue.enqueueReadBuffer(bufSAT, CL_TRUE, 0, sizeof(cl_long), &isSat);
            for (cl_long i = 0; i < bagSizeNode; i++) {
                if (sols[i] != 0) {
                    node.solution->push_back(std::pair<cl_long, solType>(i + a * bagSizeNode, sols[i]));
                    node.numSol++;
                }
            }
            delete[] sols;
        }
#ifdef DEBUG
        GPUSATUtils::printSol(node.numSol, node.numVars, node.variables, node.solution, formula);
#endif
    }

    void Solver_Incidence::solveForget(bagType &node, bagType &edge, satformulaType &formula) {
#ifdef DEBUG
        std::cout << "Solver_Incidence::solveForget:\n";
        std::cout.flush();
#endif
        auto bagSizeEdge = static_cast<cl_long>(pow(2, std::min(maxWidth, edge.numVars)));
        auto bagSizeNode = static_cast<cl_long>(pow(2, std::min(maxWidth, node.numVars)));
        cl_long numIterations = static_cast<cl_long>(pow(2, std::max((cl_long) 0, node.numVars - maxWidth)));
        node.solution = new std::vector<std::pair<cl_long, solType>>();
        node.numSol = 0;

        std::vector<cl_long> nVars;
        std::vector<cl_long> nClauses;
        for (int i = 0, a = 0, b = 0; i < node.numVars; i++) {
            if (node.variables[i] > formula.numVars) {
                nClauses.push_back(node.variables[i]);
            } else {
                nVars.push_back(node.variables[i]);
            }
        }
        nVars.push_back(0);
        nClauses.push_back(0);

        std::vector<cl_long> eVars;
        std::vector<cl_long> eClauses;
        for (int i = 0; i < edge.numVars; i++) {
            if (edge.variables[i] > formula.numVars) {
                eClauses.push_back(edge.variables[i]);
            } else {
                eVars.push_back(edge.variables[i]);
            }
        }
        eVars.push_back(0);
        eClauses.push_back(0);
        solType *eSols = new solType[bagSizeEdge]();

        for (int a = 0; a < numIterations; a++) {
            cl_long numSubIterations = static_cast<cl_long>(pow(2, std::max((cl_long) 0, edge.numVars - maxWidth)));
            cl_long d = 0;
            solType *sols = new solType[bagSizeNode]();

            for (int b = 0; b < numSubIterations; b++) {
                kernel = cl::Kernel(program, "solveForget");
                //node solutions - 0
                cl::Buffer bufNodeSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeNode), sols);
                kernel.setArg(0, bufNodeSol);
                //edge solutions - 1

                for (int i = 0; i < bagSizeEdge; i++) {
                    if (edge.solution->size() > d && edge.solution->at(d).first == b * bagSizeEdge + i) {
                        eSols[i] = edge.solution->at(d).second;
                        d++;
                    } else {
                        eSols[i] = 0.0;
                    }
                }

                cl::Buffer bufEdgeSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeEdge), eSols);
                kernel.setArg(1, bufEdgeSol);
                //node variables - 2
                cl::Buffer bufNodeVars = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * nVars.size(), &nVars[0]);
                kernel.setArg(2, bufNodeVars);
                //edge variables - 3
                cl::Buffer bufEdgeVars = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * eVars.size(), &eVars[0]);
                kernel.setArg(3, bufEdgeVars);
                //number of variables node - 4
                cl_long numNVars = nVars.size() - 1;
                cl::Buffer bufnumNVars(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &numNVars);
                kernel.setArg(4, bufnumNVars);
                //number of variables edge - 5
                cl_long numEVars = eVars.size() - 1;
                cl::Buffer bufnumEVars(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &numEVars);
                kernel.setArg(5, bufnumEVars);
                //node clauses - 6
                cl::Buffer bufNodeClauses = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * nClauses.size(), &nClauses[0]);
                kernel.setArg(6, bufNodeClauses);
                //edge clauses - 7
                cl::Buffer bufEdgeClauses = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * eClauses.size(), &eClauses[0]);
                kernel.setArg(7, bufEdgeClauses);
                //number of clauses node - 8
                cl_long numNClauses = nClauses.size() - 1;
                cl::Buffer bufNumNClauses(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &numNClauses);
                kernel.setArg(8, bufNumNClauses);
                //number of clauses edge - 9
                cl_long numEClauses = eClauses.size() - 1;
                cl::Buffer bufNumEClauses(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &numEClauses);
                kernel.setArg(9, bufNumEClauses);
                //start id node - 10
                cl_long startIdNode = a * bagSizeNode;
                cl::Buffer bufStartNode(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &startIdNode);
                kernel.setArg(10, bufStartNode);
                //start id edge - 11
                cl_long startIdEdge = b * bagSizeEdge;
                cl::Buffer bufStartEdge(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &startIdEdge);
                kernel.setArg(11, bufStartEdge);
                //min id edge - 12
                cl_long minId = b * bagSizeEdge;
                cl::Buffer bufMinId(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &minId);
                kernel.setArg(12, bufMinId);
                //max id edge - 13
                cl_long maxId = (b + 1) * bagSizeEdge;
                cl::Buffer bufMaxId(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &maxId);
                kernel.setArg(13, bufMaxId);

                queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(startIdNode)), cl::NDRange(static_cast<size_t>(bagSizeNode)));
                queue.finish();
                queue.enqueueReadBuffer(bufNodeSol, CL_TRUE, 0, sizeof(solType) * (bagSizeNode), sols);
            }
            for (cl_long i = 0; i < bagSizeNode; i++) {
                if (sols[i] != 0) {
                    node.solution->push_back(std::pair<cl_long, solType>(i + a * bagSizeNode, sols[i]));
                    node.numSol++;
                }
            }
            delete[] sols;
        }
#ifdef DEBUG
        GPUSATUtils::printSol(node.numSol, node.numVars, node.variables, node.solution, formula);
#endif
    }

    void Solver_Incidence::solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula) {
#ifdef DEBUG
        std::cout << "Solver_Incidence::solveJoin:\n";
        std::cout.flush();
#endif
        node.solution = new std::vector<std::pair<cl_long, solType>>();
        node.numSol = 0;

        bagType edge1_;
        edge1_.variables = node.variables;
        edge1_.numVars = node.numVars;
        edge1_.numSol = node.numSol;
        if (node.numVars != edge1.numVars) {
            solveIntroduce(formula, edge1_, edge1);
        } else {
            edge1_.solution = edge1.solution;
        }

        bagType edge2_;
        edge2_.variables = node.variables;
        edge2_.numVars = node.numVars;
        edge2_.numSol = node.numSol;
        if (node.numVars != edge2.numVars) {
            solveIntroduce(formula, edge2_, edge2);
        } else {
            edge2_.solution = edge2.solution;
        }

        auto bagSizeEdge1 = static_cast<cl_long>(pow(2, std::min(maxWidth, edge1_.numVars)));
        auto bagSizeEdge2 = static_cast<cl_long>(pow(2, std::min(maxWidth, edge2_.numVars)));
        auto bagSizeNode = static_cast<cl_long>(pow(2, std::min(maxWidth, node.numVars)));

        std::vector<cl_long> nClausesVector;
        std::vector<cl_long> nVars;
        for (int i = 0, a = 0, b = 0; i < node.numVars; i++) {
            if (node.variables[i] > formula.numVars) {
                nClausesVector.push_back(node.variables[i]);
            } else {
                nVars.push_back(node.variables[i]);
            }
        }
        nVars.push_back(0);
        nClausesVector.push_back(0);
        solType *eSols1 = new solType[bagSizeEdge1]();
        solType *eSols2 = new solType[bagSizeEdge2]();
        cl_long numIterations = static_cast<cl_long>(pow(2, std::max((cl_long) 0, node.numVars - maxWidth)));
        for (int a = 0; a < numIterations; a++) {
            solType *nSols = new solType[bagSizeNode]();
            cl_long numSubIterations = static_cast<cl_long>(pow(2, std::max((cl_long) 0, std::max(edge1_.numVars - maxWidth, edge2_.numVars - maxWidth))));
            cl_long d = 0, e = 0;
            for (int b = 0; b < numSubIterations; b++) {
                kernel = cl::Kernel(program, "solveJoin");
                //node solutions - 0
                cl::Buffer bufSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeNode), nSols);
                kernel.setArg(0, bufSol);
                //edge1 solutions - 1
                cl_long es1 = (b < pow(2, std::max((cl_long) 0, edge1_.numVars - maxWidth))) || b == 0 ? bagSizeEdge1 : 0;
                for (int i = 0; i < es1; i++) {
                    if (edge1_.solution->size() > d && edge1_.solution->at(d).first == bagSizeEdge1 * b + i) {
                        eSols1[i] = edge1_.solution->at(d).second;
                        d++;
                    } else {
                        eSols1[i] = 0.0;
                    }
                }
                cl::Buffer bufSol1;
                if (es1 > 0) {
                    bufSol1 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (es1),
                                         eSols1);
                    kernel.setArg(1, bufSol1);
                } else {
                    kernel.setArg(1, NULL);
                }
                //edge2 solutions - 2
                cl_long es2 = (b < pow(2, std::max((cl_long) 0, edge2_.numVars - maxWidth))) || b == 0 ? bagSizeEdge2 : 0;
                for (int i = 0; i < es2; i++) {
                    if (edge2_.solution->size() > e && edge2_.solution->at(e).first == bagSizeEdge2 * b + i) {
                        eSols2[i] = edge2_.solution->at(e).second;
                        e++;
                    } else {
                        eSols2[i] = 0.0;
                    }
                }
                cl::Buffer bufSol2;
                if (es2 > 0) {
                    bufSol2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (es2),
                                         eSols2);
                    kernel.setArg(2, bufSol2);
                } else {

                    kernel.setArg(2, NULL);
                }
                //min id edge1 - 3
                cl_long minId1 = ((b + 1) <= pow(2, std::max((cl_long) 0, edge1_.numVars - maxWidth)) || b == 0 ? bagSizeEdge1 * b : -1);
                cl::Buffer bufMinId1(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &minId1);
                kernel.setArg(3, bufMinId1);
                //max id edge1 - 4
                cl_long maxId1 = ((b + 1) <= pow(2, std::max((cl_long) 0, edge1_.numVars - maxWidth)) || b == 0 ? bagSizeEdge1 * (b + 1) : -1);
                cl::Buffer bufMaxId1(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &maxId1);
                kernel.setArg(4, bufMaxId1);
                //min id edge2 - 5
                cl_long minId2 = ((b + 1) <= pow(2, std::max((cl_long) 0, edge2_.numVars - maxWidth)) || b == 0 ? bagSizeEdge2 * b : -1);
                cl::Buffer bufMinId2(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &minId2);
                kernel.setArg(5, bufMinId2);
                //max id edge2 - 6
                cl_long maxId2 = ((b + 1) <= pow(2, std::max((cl_long) 0, edge2_.numVars - maxWidth)) || b == 0 ? bagSizeEdge2 * (b + 1) : -1);
                cl::Buffer bufMaxId2(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &maxId2);
                kernel.setArg(6, bufMaxId2);
                //start id node - 7
                cl_long startIdNode = a * bagSizeNode;
                cl::Buffer bufStartNode(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &startIdNode);
                kernel.setArg(7, bufStartNode);
                //start id edge1 - 8
                cl_long startIdEdge1 = b * bagSizeEdge1;
                cl::Buffer bufStartEdge1(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &startIdEdge1);
                kernel.setArg(8, bufStartEdge1);
                //start id edge2 - 9
                cl_long startIdEdge2 = b * bagSizeEdge2;
                cl::Buffer bufStartEdge2(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &startIdEdge2);
                kernel.setArg(9, bufStartEdge2);
                //number of clauses - 10
                cl_long numClauses = nClausesVector.size() - 1;
                cl::Buffer bufNumClauses(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &numClauses);
                kernel.setArg(10, bufNumClauses);
                //weights - 11
                cl::Buffer bufWeights;
                if (formula.variableWeights != nullptr) {
                    bufWeights = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * formula.numWeights, formula.variableWeights);
                    kernel.setArg(11, bufWeights);
                } else {
                    kernel.setArg(11, NULL);
                }
                //node variables - 12
                cl::Buffer bufVariables = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (nVars.size()), &nVars[0]);
                kernel.setArg(12, bufVariables);

                queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(startIdNode)), cl::NDRange(static_cast<size_t>(bagSizeNode)));
                queue.finish();
                queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(solType) * (bagSizeNode), nSols);
            }
            for (cl_long i = 0; i < bagSizeNode; i++) {
                if (nSols[i] != 0) {
                    node.solution->push_back(std::pair<cl_long, solType>(i + a * bagSizeNode, nSols[i]));
                    node.numSol++;
                }
            }
            delete[] nSols;
        }
#ifdef DEBUG
        GPUSATUtils::printSol(node.numSol, node.numVars, node.variables, node.solution, formula);
#endif
    }
}