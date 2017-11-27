#include <algorithm>
#include <cmath>
#include <iostream>
#include <d4_utils.h>
#include <gpusautils.h>
#include <solver.h>

namespace gpusat {
    void Solver::solveProblem(treedecType &decomp, satformulaType &formula, bagType &node, bagType &pnode) {
        if (isSat > 0) {
            if (node.edges.size() == 0) {
                bagType cNode;
                cNode.numSol = 1;
                cNode.solution = new solType *[1];
                cNode.solution[0] = new solType[1];
                cNode.solution[0][0] = 1.0;
                solveIntroduceForget(formula, pnode, node, cNode);
            } else if (node.edges.size() == 1) {
                cl_long edge = node.edges[0] - 1;
                solveProblem(decomp, formula, decomp.bags[edge], node);
                if (isSat == 1) {
                    bagType &cnode = decomp.bags[node.edges[0] - 1];
                    solveIntroduceForget(formula, pnode, node, cnode);
                }
            } else if (node.edges.size() > 1) {
                cl_long edge = node.edges[0] - 1;
                bagType &edge1 = decomp.bags[edge];
                solveProblem(decomp, formula, edge1, node);
                if (isSat <= 0) {
                    return;
                }
                if (isSat == 1) {
                    bagType tmp, edge2_, edge1_;

                    for (int i = 1; i < node.edges.size(); i++) {
                        edge = node.edges[i] - 1;
                        bagType &edge2 = decomp.bags[edge];
                        solveProblem(decomp, formula, edge2, node);
                        if (isSat <= 0) {
                            return;
                        }

                        std::vector<cl_long> vt(static_cast<unsigned long long int>(edge1.variables.size() + edge2.variables.size()));
                        auto itt = std::set_union(edge1.variables.begin(), edge1.variables.end(), edge2.variables.begin(), edge2.variables.end(), vt.begin());
                        vt.resize(static_cast<unsigned long long int>(itt - vt.begin()));
                        tmp.variables = vt;
                        tmp.numSol = static_cast<cl_long>(pow(2, tmp.variables.size()));
                        solveJoin(tmp, edge1, edge2, formula);
                        edge1 = tmp;

                        if (i == node.edges.size() - 1) {
                            solveIntroduceForget(formula, pnode, node, tmp);
                        }
                    }
                }
            }
        }
    }

    void Solver_Primal::solveIntroduceForget(satformulaType &formula, bagType &pnode, bagType &node, bagType &cnode) {
        isSat = 0;
        std::vector<cl_long> fVars;
        std::set_intersection(node.variables.begin(), node.variables.end(), pnode.variables.begin(), pnode.variables.end(), std::back_inserter(fVars));
        std::vector<cl_long> iVars = node.variables;
        std::vector<cl_long> eVars = cnode.variables;

        node.variables = fVars;
        node.numSol = pow(2, node.variables.size());

        cl_long bagSizeEdge = static_cast<cl_long>(pow(2, std::min((cl_long) eVars.size(), maxWidth)));
        cl_long bagSizeForget = static_cast<cl_long>(pow(2, std::min((cl_long) fVars.size(), maxWidth)));

        cl_long numIterations = (cl_long) ceil(node.numSol / bagSizeForget);
        cl_long numSubIterations = (cl_long) ceil(cnode.numSol / bagSizeEdge);

        node.solution = new solType *[(cl_long) ceil(node.numSol / bagSizeForget)];
        cl_long numHpath = pow(2, fVars.size());
        this->numForget++;

        std::vector<cl_long> numVarsClause;
        std::vector<cl_long> clauses;
        cl_long numClauses = 0;
        for (cl_long i = 0; i < formula.clauses.size(); i++) {
            std::vector<cl_long> v(formula.clauses[i].size());
            std::vector<cl_long>::iterator it;
            it = std::set_intersection(iVars.begin(), iVars.end(), formula.clauses[i].begin(), formula.clauses[i].end(),
                                       v.begin(), compVars);
            if (it - v.begin() == formula.clauses[i].size()) {
                numClauses++;
                numVarsClause.push_back(formula.clauses[i].size());
                for (cl_long a = 0; a < formula.clauses[i].size(); a++) {
                    clauses.push_back(formula.clauses[i][a]);
                }
            }
        }

        for (int a = 0; a < numIterations; a++) {
            cl_int solutions = 0;
            node.solution[a] = new solType[bagSizeForget]();
            cl_long startIdNode = a * bagSizeForget;
            kernel = cl::Kernel(program, "solveIntroduceForget");
            cl::Buffer buf_solsF(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeForget), node.solution[a]);
            kernel.setArg(0, buf_solsF);
            cl::Buffer buf_varsF;
            if (fVars.size() > 0) {
                buf_varsF = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * fVars.size(), &fVars[0]);
                kernel.setArg(1, buf_varsF);
            } else {
                kernel.setArg(1, NULL);
            }
            for (int b = 0; b < numSubIterations; b++) {
                if (cnode.solution[b] == nullptr) {
                    continue;
                }
                cl::Buffer buf_solsE(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeEdge), cnode.solution[b]);
                kernel.setArg(2, buf_solsE);

                kernel.setArg(3, eVars.size());

                cl::Buffer buf_varsE;
                if (eVars.size() > 0) {
                    buf_varsE = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * eVars.size(), &eVars[0]);
                    kernel.setArg(4, buf_varsE);
                } else {
                    kernel.setArg(4, NULL);
                }

                kernel.setArg(5, (cl_long) pow(2, iVars.size() - fVars.size()));
                kernel.setArg(6, fVars.size());
                cl_long minId = b * bagSizeEdge;
                kernel.setArg(7, minId);
                cl_long maxId = (b + 1) * bagSizeEdge;
                kernel.setArg(8, maxId);
                kernel.setArg(9, startIdNode);
                cl_long startIdEdge = b * bagSizeEdge;
                kernel.setArg(10, startIdEdge);

                cl_int bagsolutions = 0;
                cl::Buffer bufsolBag(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &bagsolutions);
                kernel.setArg(11, bufsolBag);

                //number Variables introduce
                kernel.setArg(12, iVars.size());

                //variables introduce
                cl::Buffer bufIVars;
                if (iVars.size() > 0) {
                    bufIVars = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * iVars.size(), &iVars[0]);
                    kernel.setArg(13, bufIVars);
                } else {
                    kernel.setArg(13, NULL);
                }
                cl::Buffer bufClauses;
                if (clauses.size() > 0) {
                    bufClauses = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (clauses.size()), &clauses[0]);
                    kernel.setArg(14, bufClauses);
                } else {
                    kernel.setArg(14, NULL);
                }
                cl::Buffer bufNumVarsC;
                if (numClauses > 0) {
                    bufNumVarsC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (numClauses), &numVarsClause[0]);
                    kernel.setArg(15, bufNumVarsC);
                } else {
                    kernel.setArg(15, NULL);
                }
                kernel.setArg(16, numClauses);
                cl::Buffer bufWeights;
                if (formula.variableWeights != nullptr) {
                    bufWeights = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * formula.numWeights, formula.variableWeights);
                    kernel.setArg(17, bufWeights);
                } else {
                    kernel.setArg(17, NULL);
                }

                queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(startIdNode)), cl::NDRange(static_cast<size_t>(bagSizeForget)));
                queue.finish();
                queue.enqueueReadBuffer(bufsolBag, CL_TRUE, 0, sizeof(cl_int), &bagsolutions);
                solutions += bagsolutions;
                queue.enqueueReadBuffer(buf_solsF, CL_TRUE, 0, sizeof(solType) * (bagSizeForget), node.solution[a]);
            }
            if (solutions == 0) {
                delete[] node.solution[a];
                node.solution[a] = nullptr;
                numHpath -= bagSizeForget;
            } else {
                this->isSat = 1;
            }
        }
        for (cl_long a = 0; a < cnode.numSol / bagSizeEdge; a++) {
            if (cnode.solution[a] != nullptr) {
                delete[] cnode.solution[a];
                cnode.solution[a] = nullptr;
            }
        }
        if (this->getStats) {
            cl_long numSpath = 0;
            this->numHoldPaths.push_back(numHpath);
            for (cl_long a = 0; a < numIterations; a++) {
                if (node.solution[a] != nullptr) {
                    for (cl_long b = 0; b < bagSizeForget; b++) {
                        if (node.solution[a][b] > 0) {
                            numSpath++;
                        }
                    }
                }
            }
            this->numSolPaths.push_back(numSpath);
        }
#ifdef DEBUG
        std::cout << "solveIntroduceForget:\n";
        GPUSATUtils::printSol(node.numSol, node.variables, node.solution, formula, bagSizeForget);
#endif
    }

    void Solver_Incidence::solveIntroduceForget(satformulaType &formula, bagType &pnode, bagType &node, bagType &cnode) {
        isSat = 0;
        std::vector<cl_long> fNode;
        std::set_intersection(node.variables.begin(), node.variables.end(), pnode.variables.begin(), pnode.variables.end(), std::back_inserter(fNode));
        std::vector<cl_long> iNode = node.variables;
        std::vector<cl_long> eNode = cnode.variables;

        node.variables = fNode;
        node.numSol = pow(2, node.variables.size());

        auto bagSizeEdge = static_cast<cl_long>(pow(2, std::min(maxWidth, (cl_long) eNode.size())));
        auto bagSizeNode = static_cast<cl_long>(pow(2, std::min(maxWidth, (cl_long) node.variables.size())));
        cl_long numIterations = static_cast<cl_long>(pow(2, std::max((cl_long) 0, (cl_long) node.variables.size() - maxWidth)));
        node.solution = new solType *[(cl_long) ceil(node.numSol / bagSizeNode)];
        cl_long numHpath = pow(2, node.variables.size());
        this->numForget++;

        std::vector<cl_long> fVars;
        std::vector<cl_long> fClauses;
        for (int i = 0; i < fNode.size(); i++) {
            if (fNode[i] > formula.numVars) {
                fClauses.push_back(fNode[i]);
            } else {
                fVars.push_back(fNode[i]);
            }
        }
        fVars.push_back(0);
        fClauses.push_back(0);

        std::vector<cl_long> iVars;
        std::vector<cl_long> iClauses;
        for (int i = 0; i < iNode.size(); i++) {
            if (iNode[i] > formula.numVars) {
                iClauses.push_back(iNode[i]);
            } else {
                iVars.push_back(iNode[i]);
            }
        }
        iVars.push_back(0);
        iClauses.push_back(0);

        std::vector<cl_long> eVars;
        std::vector<cl_long> eClauses;
        for (int i = 0; i < eNode.size(); i++) {
            if (eNode[i] > formula.numVars) {
                eClauses.push_back(eNode[i]);
            } else {
                eVars.push_back(eNode[i]);
            }
        }
        eVars.push_back(0);
        eClauses.push_back(0);

        std::vector<cl_long> clauseVector;
        for (int i = 0; i < iNode.size(); i++) {
            if (iNode[i] > formula.numVars) {
                for (cl_long a = 0; a < formula.clauses[iNode[i] - formula.numVars - 1].size(); a++) {
                    clauseVector.push_back(formula.clauses[iNode[i] - formula.numVars - 1][a]);
                }
                clauseVector.push_back(0);
            }
        }

        for (int a = 0; a < numIterations; a++) {
            cl_int solutions = 0;
            node.solution[a] = new solType[bagSizeNode]();
            cl_long numSubIterations = static_cast<cl_long>(pow(2, std::max((cl_long) 0, (cl_long) eNode.size() - maxWidth)));
            kernel = cl::Kernel(program, "solveIntroduceForget");
            //node solutions - 0
            cl::Buffer bufNodeSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeNode), node.solution[a]);
            kernel.setArg(0, bufNodeSol);
            for (int b = 0; b < numSubIterations; b++) {
                if (cnode.solution[b] == nullptr) {
                    continue;
                }
                //edge solutions - 1
                cl::Buffer bufEdgeSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeEdge), cnode.solution[b]);
                kernel.setArg(1, bufEdgeSol);
                //node variables - 2
                cl::Buffer bufNodeVars = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * fVars.size(), &fVars[0]);
                kernel.setArg(2, bufNodeVars);
                //edge variables - 3
                cl::Buffer bufEdgeVars = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * eVars.size(), &eVars[0]);
                kernel.setArg(3, bufEdgeVars);
                //number of variables node - 4
                kernel.setArg(4, fVars.size() - 1);
                //number of variables edge - 5
                kernel.setArg(5, eVars.size() - 1);
                //node clauses - 6
                cl::Buffer bufNodeClauses = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * fClauses.size(), &fClauses[0]);
                kernel.setArg(6, bufNodeClauses);
                //edge clauses - 7
                cl::Buffer bufEdgeClauses = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * eClauses.size(), &eClauses[0]);
                kernel.setArg(7, bufEdgeClauses);
                //number of clauses node - 8
                kernel.setArg(8, fClauses.size() - 1);
                //number of clauses edge - 9
                kernel.setArg(9, eClauses.size() - 1);
                //start id node - 10
                cl_long startIdNode = a * bagSizeNode;
                kernel.setArg(10, startIdNode);
                //start id edge - 11
                cl_long startIdEdge = b * bagSizeEdge;
                kernel.setArg(11, startIdEdge);
                //min id edge - 12
                cl_long minId = b * bagSizeEdge;
                kernel.setArg(12, minId);
                //max id edge - 13
                cl_long maxId = (b + 1) * bagSizeEdge;
                kernel.setArg(13, maxId);
                cl_int bagsolutions = 0;
                cl::Buffer bufsolBag(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &bagsolutions);
                kernel.setArg(14, bufsolBag);


                cl::Buffer bufIVars = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * iVars.size(), &iVars[0]);
                kernel.setArg(15, bufIVars);
                kernel.setArg(16, iVars.size() - 1);
                cl::Buffer bufIClauses = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * iClauses.size(), &iClauses[0]);
                kernel.setArg(17, bufIClauses);
                kernel.setArg(18, iClauses.size() - 1);
                //clauses - 2
                cl::Buffer bufClauses = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (clauseVector.size()),
                                                   &clauseVector[0]);
                kernel.setArg(19, bufClauses);
                //clauses array length - 3
                cl_long clausesLength = clauseVector.size();
                kernel.setArg(20, clausesLength);
                cl::Buffer bufWeights;
                if (formula.variableWeights != nullptr) {
                    bufWeights = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * formula.numWeights, formula.variableWeights);
                    kernel.setArg(21, bufWeights);
                } else {
                    kernel.setArg(21, NULL);
                }

                queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(startIdNode)), cl::NDRange(static_cast<size_t>(bagSizeNode)));
                queue.finish();
                queue.enqueueReadBuffer(bufsolBag, CL_TRUE, 0, sizeof(cl_int), &bagsolutions);
                queue.enqueueReadBuffer(bufNodeSol, CL_TRUE, 0, sizeof(solType) * (bagSizeNode), node.solution[a]);
                if (bagsolutions != 0) {
                    solutions = 1;
                    isSat = 1;
                }
            }
            if (solutions == 0) {
                delete[] node.solution[a];
                node.solution[a] = nullptr;
                numHpath -= bagSizeNode;
            }
        }
        for (cl_long a = 0; a < cnode.numSol / bagSizeEdge; a++) {
            if (cnode.solution[a] != nullptr) {
                delete[] cnode.solution[a];
                cnode.solution[a] = nullptr;
            }
        }
        if (this->getStats) {
            cl_long numSpath = 0;
            this->numHoldPaths.push_back(numHpath);
            for (cl_long a = 0; a < numIterations; a++) {
                if (node.solution[a] != nullptr) {
                    for (cl_long b = 0; b < bagSizeNode; b++) {
                        if (node.solution[a][b] > 0) {
                            numSpath++;
                        }
                    }
                }
            }
            this->numSolPaths.push_back(numSpath);
        }
#ifdef DEBUG
        std::cout << "solveIntroduceForget:\n";
        GPUSATUtils::printSol(node.numSol, node.variables, node.solution, formula, bagSizeNode);
#endif
    }

    void Solver_Primal::solveIntroduce(satformulaType &formula, bagType &node, bagType &edge) {
        cl_long bagSizeEdge = static_cast<cl_long>(pow(2, std::min(maxWidth, (cl_long) edge.variables.size())));
        cl_long bagSizeNode = static_cast<cl_long>(pow(2, std::min(maxWidth, (cl_long) node.variables.size())));
        cl_long numIterations = (cl_long) ceil(node.numSol / bagSizeNode);
        cl_long numSubIterations = (cl_long) ceil(edge.numSol / bagSizeEdge);
        node.solution = new solType *[(cl_long) ceil(node.numSol / bagSizeNode)];
        isSat = 0;
        cl_long numHpath = pow(2, node.variables.size());
        this->numIntroduce++;

        // get clauses
        std::vector<cl_long> numVarsClause;
        std::vector<cl_long> clauses;
        cl_long numClauses = 0;
        for (cl_long i = 0; i < formula.clauses.size(); i++) {
            std::vector<cl_long> v(formula.clauses[i].size());
            std::vector<cl_long>::iterator it;
            it = std::set_intersection(node.variables.begin(), node.variables.end(), formula.clauses[i].begin(), formula.clauses[i].end(),
                                       v.begin(), compVars);
            if (it - v.begin() == formula.clauses[i].size()) {
                numClauses++;
                numVarsClause.push_back(formula.clauses[i].size());
                for (cl_long a = 0; a < formula.clauses[i].size(); a++) {
                    clauses.push_back(formula.clauses[i][a]);
                }
            }
        }

        for (int a = 0; a < numIterations; a++) {
            cl_int solutions = 0;
            node.solution[a] = new solType[bagSizeNode]();
            cl_long startIdNode = a * bagSizeNode;
            kernel = cl::Kernel(program, "solveIntroduce");
            cl::Buffer bufSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeNode), node.solution[a]);
            kernel.setArg(3, bufSol);
            for (int b = 0; b < numSubIterations; b++) {
                if (edge.solution[b] == nullptr) {
                    continue;
                }
                cl_long startIdEdge = b * bagSizeEdge;
                cl::Buffer bufVertices;
                if (node.variables.size() > 0) {
                    bufVertices = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (node.variables.size()), &node.variables[0]);
                    kernel.setArg(7, bufVertices);
                } else {
                    kernel.setArg(7, NULL);
                }
                cl::Buffer bufClauses;
                if (clauses.size() > 0) {
                    bufClauses = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (clauses.size()), &clauses[0]);
                    kernel.setArg(0, bufClauses);
                } else {
                    kernel.setArg(0, NULL);
                }
                cl::Buffer bufNumVarsC;
                if (numClauses > 0) {
                    bufNumVarsC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (numClauses), &numVarsClause[0]);
                    kernel.setArg(1, bufNumVarsC);
                } else {
                    kernel.setArg(1, NULL);
                }
                cl::Buffer bufSolNext(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeEdge), edge.solution[b]);
                cl::Buffer bufNextVars;
                if (edge.variables.size() > 0) {
                    bufNextVars = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (edge.variables.size()), &edge.variables[0]);
                    kernel.setArg(8, bufNextVars);
                } else {
                    kernel.setArg(8, NULL);
                }
                cl::Buffer bufSAT(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &isSat);
                kernel.setArg(2, numClauses);
                kernel.setArg(4, node.variables.size());
                kernel.setArg(5, bufSolNext);
                kernel.setArg(6, edge.variables.size());
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
                cl_int bagsolutions = 0;
                cl::Buffer bufsolBag(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &bagsolutions);
                kernel.setArg(15, bufsolBag);

                queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(startIdNode)), cl::NDRange(static_cast<size_t>(bagSizeNode)));
                queue.finish();
                queue.enqueueReadBuffer(bufSAT, CL_TRUE, 0, sizeof(cl_long), &isSat);
                queue.enqueueReadBuffer(bufsolBag, CL_TRUE, 0, sizeof(cl_int), &bagsolutions);
                queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(solType) * (bagSizeNode), node.solution[a]);
                solutions += bagsolutions;
            }
            if (solutions == 0) {
                delete[] node.solution[a];
                node.solution[a] = nullptr;
                numHpath -= bagSizeNode;
            }
        }
        for (cl_long a = 0; a < edge.numSol / bagSizeEdge; a++) {
            if (edge.solution[a] != nullptr) {
                delete[] edge.solution[a];
                edge.solution[a] = nullptr;
            }
        }
        if (this->getStats) {
            cl_long numSpath = 0;
            this->numHoldPaths.push_back(numHpath);
            for (cl_long a = 0; a < numIterations; a++) {
                if (node.solution[a] != nullptr) {
                    for (cl_long b = 0; b < bagSizeNode; b++) {
                        if (node.solution[a][b] > 0) {
                            numSpath++;
                        }
                    }
                }
            }
            this->numSolPaths.push_back(numSpath);
        }
#ifdef DEBUG
        std::cout << "Introduce:\n";
        GPUSATUtils::printSol(node.numSol, node.variables, node.solution, formula, bagSizeNode);
#endif
    }

    void Solver_Primal::solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula) {
        cl_long bagSizeEdge1 = static_cast<cl_long>(pow(2, std::min(maxWidth, (cl_long) edge1.variables.size())));
        cl_long bagSizeEdge2 = static_cast<cl_long>(pow(2, std::min(maxWidth, (cl_long) edge2.variables.size())));
        cl_long bagSizeNode = static_cast<cl_long>(pow(2, std::min(maxWidth, (cl_long) node.variables.size())));
        node.solution = new solType *[(cl_long) ceil(node.numSol / bagSizeNode)];
        cl_long numIterations = (cl_long) ceil(node.numSol / bagSizeNode);
        cl_long numIterationsEdge1 = (cl_long) ceil(edge1.numSol / bagSizeEdge1);
        cl_long numIterationsEdge2 = (cl_long) ceil(edge2.numSol / bagSizeEdge2);
        cl_long numHpath = pow(2, node.variables.size());
        this->numJoin++;
        for (int a = 0; a < numIterations; a++) {
            cl_int solutions = 0;
            node.solution[a] = new solType[bagSizeNode]();
            for (cl_long i = 0; i < bagSizeNode; i++) {
                node.solution[a][i] = 1.0;
            }
            cl_long startIdNode = a * bagSizeNode;
            cl_long startIdEdge1;
            cl_long startIdEdge2;
            kernel = cl::Kernel(program, "solveJoin");
            cl::Buffer bufSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeNode), node.solution[a]);
            kernel.setArg(0, bufSol);
            cl::Buffer bufSolVars;
            if (node.variables.size() > 0) {
                bufSolVars = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * node.variables.size(), &node.variables[0]);
                kernel.setArg(3, bufSolVars);
            } else {
                kernel.setArg(3, NULL);
            }
            for (int b = 0; b < std::max(numIterationsEdge1, numIterationsEdge2); b++) {
                startIdEdge1 = b * bagSizeEdge1;
                startIdEdge2 = b * bagSizeEdge2;
                cl::Buffer bufSol1;
                if (b < numIterationsEdge1 && edge1.solution[b] != nullptr) {
                    bufSol1 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeEdge1), edge1.solution[b]);
                    kernel.setArg(1, bufSol1);
                } else {
                    kernel.setArg(1, NULL);
                }
                cl::Buffer bufSol2;
                if (b < numIterationsEdge2 && edge2.solution[b] != nullptr) {
                    bufSol2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeEdge2), edge2.solution[b]);
                    kernel.setArg(2, bufSol2);
                } else {
                    kernel.setArg(2, NULL);
                }
                cl::Buffer bufSolVars1;
                if (edge1.variables.size() > 0) {
                    bufSolVars1 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * edge1.variables.size(), &edge1.variables[0]);
                    kernel.setArg(4, bufSolVars1);
                } else {
                    kernel.setArg(4, NULL);
                }
                cl::Buffer bufSolVars2;
                if (edge2.variables.size() > 0) {
                    bufSolVars2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * edge2.variables.size(), &edge2.variables[0]);
                    kernel.setArg(5, bufSolVars2);
                } else {
                    kernel.setArg(5, NULL);
                }
                kernel.setArg(6, node.variables.size());
                kernel.setArg(7, edge1.variables.size());
                kernel.setArg(8, edge2.variables.size());
                cl_long minId1 = ((b + 1) <= pow(2, std::max((cl_long) 0, (cl_long) edge1.variables.size() - maxWidth)) || b == 0 ? bagSizeEdge1 * b : -1);
                cl_long maxId1 = ((b + 1) <= pow(2, std::max((cl_long) 0, (cl_long) edge1.variables.size() - maxWidth)) || b == 0 ? bagSizeEdge1 * (b + 1) : -1);
                cl::Buffer bufMinId1(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &minId1);
                cl::Buffer bufMaxId1(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &maxId1);
                kernel.setArg(9, bufMinId1);
                kernel.setArg(10, bufMaxId1);
                cl_long minId2 = ((b + 1) <= pow(2, std::max((cl_long) 0, (cl_long) edge2.variables.size() - maxWidth)) || b == 0 ? bagSizeEdge2 * b : -1);
                cl_long maxId2 = ((b + 1) <= pow(2, std::max((cl_long) 0, (cl_long) edge2.variables.size() - maxWidth)) || b == 0 ? bagSizeEdge2 * (b + 1) : -1);
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
                cl_int bagsolutions = 0;
                cl::Buffer bufsolBag(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &bagsolutions);
                kernel.setArg(17, bufsolBag);

                queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(startIdNode)), cl::NDRange(static_cast<size_t>(bagSizeNode)));
                queue.finish();
                queue.enqueueReadBuffer(bufsolBag, CL_TRUE, 0, sizeof(cl_int), &bagsolutions);
                solutions += bagsolutions;
                queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(solType) * (bagSizeNode), node.solution[a]);
            }
            if (solutions == 0) {
                delete[] node.solution[a];
                node.solution[a] = nullptr;
                numHpath -= bagSizeNode;
            }
        }
        for (cl_long a = 0; a < edge1.numSol / bagSizeEdge1; a++) {
            if (edge1.solution[a] != nullptr) {
                delete[] edge1.solution[a];
                edge1.solution[a] = nullptr;
            }
        }
        for (cl_long a = 0; a < edge2.numSol / bagSizeEdge2; a++) {
            if (edge2.solution[a] != nullptr) {
                delete[] edge2.solution[a];
                edge2.solution[a] = nullptr;
            }
        }
        if (this->getStats) {
            cl_long numSpath = 0;
            this->numHoldPaths.push_back(numHpath);
            for (cl_long a = 0; a < numIterations; a++) {
                if (node.solution[a] != nullptr) {
                    for (cl_long b = 0; b < bagSizeNode; b++) {
                        if (node.solution[a][b] > 0) {
                            numSpath++;
                        }
                    }
                }
            }
            this->numSolPaths.push_back(numSpath);
        }
#ifdef DEBUG
        std::cout << "Join:\n";
        GPUSATUtils::printSol(node.numSol, node.variables, node.solution, formula, bagSizeNode);
#endif
    }

    void Solver_Incidence::solveIntroduce(satformulaType &formula, bagType &node, bagType &edge) {
        cl_long bagSizeEdge = static_cast<cl_long>(pow(2, std::min(maxWidth, (cl_long) edge.variables.size())));
        cl_long bagSizeNode = static_cast<cl_long>(pow(2, std::min(maxWidth, (cl_long) node.variables.size())));
        cl_long numIterations = static_cast<cl_long>(pow(2, std::max((cl_long) 0, (cl_long) node.variables.size() - maxWidth)));
        node.solution = new solType *[(cl_long) ceil(node.numSol / bagSizeNode)];
        isSat = 0;
        cl_long numHpath = pow(2, node.variables.size());
        this->numIntroduce++;

        std::vector<cl_long> clauseVector;
        for (int i = 0, a = 0, b = 0; i < node.variables.size(); i++) {
            if (node.variables[i] > formula.numVars) {
                for (cl_long a = 0; a < formula.clauses[node.variables[i] - formula.numVars - 1].size(); a++) {
                    clauseVector.push_back(formula.clauses[node.variables[i] - formula.numVars - 1][a]);
                }
                clauseVector.push_back(0);
            }
        }

        std::vector<cl_long> nVars;
        std::vector<cl_long> nClauses;
        for (int i = 0; i < node.variables.size(); i++) {
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
        for (int i = 0; i < edge.variables.size(); i++) {
            if (edge.variables[i] > formula.numVars) {
                eClauses.push_back(edge.variables[i]);
            } else {
                eVars.push_back(edge.variables[i]);
            }
        }
        eVars.push_back(0);
        eClauses.push_back(0);
        cl_long numSubIterations = static_cast<cl_long>(pow(2, std::max((cl_long) 0, (cl_long) edge.variables.size() - maxWidth)));

        for (int a = 0; a < numIterations; a++) {
            cl_int solutions = 0;
            node.solution[a] = new solType[bagSizeNode]();
            kernel = cl::Kernel(program, "solveIntroduce");
            //node solutions - 0
            cl::Buffer bufNSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeNode), node.solution[a]);
            kernel.setArg(0, bufNSol);
            for (int b = 0; b < numSubIterations; b++) {
                if (edge.solution[b] == nullptr) {
                    continue;
                }
                //edge solutions - 1
                cl::Buffer bufESol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeEdge), edge.solution[b]);
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
                cl_int bagsolutions = 0;
                cl::Buffer bufsolBag(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &bagsolutions);
                kernel.setArg(18, bufsolBag);

                queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(startIdNode)), cl::NDRange(static_cast<size_t>(bagSizeNode)));
                queue.finish();
                queue.enqueueReadBuffer(bufSAT, CL_TRUE, 0, sizeof(cl_long), &isSat);
                queue.enqueueReadBuffer(bufSAT, CL_TRUE, 0, sizeof(cl_int), &bagsolutions);
                queue.enqueueReadBuffer(bufNSol, CL_TRUE, 0, sizeof(solType) * (bagSizeNode), node.solution[a]);
                if (bagsolutions != 0) {
                    solutions = 1;
                }
            }
            if (solutions == 0) {
                delete[] node.solution[a];
                node.solution[a] = nullptr;
                numHpath -= bagSizeNode;
            }
        }
        for (cl_long a = 0; a < edge.numSol / bagSizeEdge; a++) {
            if (edge.solution[a] != nullptr) {
                delete[] edge.solution[a];
                edge.solution[a] = nullptr;
            }
        }
        if (this->getStats) {
            cl_long numSpath = 0;
            this->numHoldPaths.push_back(numHpath);
            for (cl_long a = 0; a < numIterations; a++) {
                if (node.solution[a] != nullptr) {
                    for (cl_long b = 0; b < bagSizeNode; b++) {
                        if (node.solution[a][b] > 0) {
                            numSpath++;
                        }
                    }
                }
            }
            this->numSolPaths.push_back(numSpath);
        }
#ifdef DEBUG
        std::cout << "Introduce:\n";
        GPUSATUtils::printSol(node.numSol, node.variables, node.solution, formula, bagSizeNode);
#endif
    }

    void Solver_Incidence::solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula) {
        bagType edge1_;
        cl_long numHpath = pow(2, node.variables.size());
        this->numJoin++;
        edge1_.variables = node.variables;
        edge1_.numSol = node.numSol;
        if (node.variables.size() != edge1.variables.size()) {
            solveIntroduce(formula, edge1_, edge1);
        } else {
            edge1_.solution = edge1.solution;
        }

        bagType edge2_;
        edge2_.variables = node.variables;
        edge2_.numSol = node.numSol;
        if (node.variables.size() != edge2.variables.size()) {
            solveIntroduce(formula, edge2_, edge2);
        } else {
            edge2_.solution = edge2.solution;
        }

        auto bagSizeEdge1 = static_cast<cl_long>(pow(2, std::min(maxWidth, (cl_long) edge1_.variables.size())));
        auto bagSizeEdge2 = static_cast<cl_long>(pow(2, std::min(maxWidth, (cl_long) edge2_.variables.size())));
        auto bagSizeNode = static_cast<cl_long>(pow(2, std::min(maxWidth, (cl_long) node.variables.size())));
        node.solution = new solType *[(cl_long) ceil(node.numSol / bagSizeNode)];

        std::vector<cl_long> nClausesVector;
        std::vector<cl_long> nVars;
        for (int i = 0, a = 0, b = 0; i < node.variables.size(); i++) {
            if (node.variables[i] > formula.numVars) {
                nClausesVector.push_back(node.variables[i]);
            } else {
                nVars.push_back(node.variables[i]);
            }
        }
        nVars.push_back(0);
        nClausesVector.push_back(0);

        cl_long numIterations = static_cast<cl_long>(pow(2, std::max((cl_long) 0, (cl_long) node.variables.size() - maxWidth)));
        for (int a = 0; a < numIterations; a++) {
            cl_int solutions = 0;
            node.solution[a] = new solType[bagSizeNode]();
            kernel = cl::Kernel(program, "solveJoin");
            //node solutions - 0
            cl::Buffer bufSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (bagSizeNode), node.solution[a]);
            kernel.setArg(0, bufSol);
            for (int b = 0; b < numIterations; b++) {
                if (edge1_.solution[b] == nullptr) {
                    continue;
                }
                //edge1 solutions - 1
                cl_long es1 = (b < pow(2, std::max((cl_long) 0, (cl_long) edge1_.variables.size() - maxWidth))) || b == 0 ? bagSizeEdge1 : 0;
                cl::Buffer bufSol1;
                if (es1 > 0) {
                    bufSol1 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (es1),
                                         edge1_.solution[b]);
                    kernel.setArg(1, bufSol1);
                } else {
                    kernel.setArg(1, NULL);
                }
                for (int c = 0; c < numIterations; c++) {
                    if (edge2_.solution[c] == nullptr) {
                        continue;
                    }
                    //edge2 solutions - 2
                    cl_long es2 = (c < pow(2, std::max((cl_long) 0, (cl_long) edge2_.variables.size() - maxWidth))) || c == 0 ? bagSizeEdge2 : 0;
                    cl::Buffer bufSol2;
                    if (es2 > 0) {
                        bufSol2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(solType) * (es2),
                                             edge2_.solution[c]);
                        kernel.setArg(2, bufSol2);
                    } else {

                        kernel.setArg(2, NULL);
                    }
                    //min id edge1 - 3
                    cl_long minId1 = ((b + 1) <= pow(2, std::max((cl_long) 0, (cl_long) edge1_.variables.size() - maxWidth)) || b == 0 ? bagSizeEdge1 * b : -1);
                    cl::Buffer bufMinId1(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &minId1);
                    kernel.setArg(3, bufMinId1);
                    //max id edge1 - 4
                    cl_long maxId1 = ((b + 1) <= pow(2, std::max((cl_long) 0, (cl_long) edge1_.variables.size() - maxWidth)) || b == 0 ? bagSizeEdge1 * (b + 1) : -1);
                    cl::Buffer bufMaxId1(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &maxId1);
                    kernel.setArg(4, bufMaxId1);
                    //min id edge2 - 5
                    cl_long minId2 = ((c + 1) <= pow(2, std::max((cl_long) 0, (cl_long) edge2_.variables.size() - maxWidth)) || c == 0 ? bagSizeEdge2 * c : -1);
                    cl::Buffer bufMinId2(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &minId2);
                    kernel.setArg(5, bufMinId2);
                    //max id edge2 - 6
                    cl_long maxId2 = ((c + 1) <= pow(2, std::max((cl_long) 0, (cl_long) edge2_.variables.size() - maxWidth)) || c == 0 ? bagSizeEdge2 * (c + 1) : -1);
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
                    cl_long startIdEdge2 = c * bagSizeEdge2;
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
                    cl_int bagsolutions = 0;
                    cl::Buffer bufsolBag(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &bagsolutions);
                    kernel.setArg(13, bufsolBag);

                    queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(startIdNode)), cl::NDRange(static_cast<size_t>(bagSizeNode)));
                    queue.finish();
                    queue.enqueueReadBuffer(bufsolBag, CL_TRUE, 0, sizeof(cl_int), &bagsolutions);
                    if (bagsolutions > 0) {
                        solutions = 1;
                    }
                    queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(solType) * (bagSizeNode), node.solution[a]);
                }
            }
            if (solutions == 0) {
                delete[] node.solution[a];
                node.solution[a] = nullptr;
                numHpath -= bagSizeNode;
            }
        }
        if (node.variables.size() != edge1.variables.size()) {
            for (cl_long a = 0; a < edge1_.numSol / bagSizeEdge1; a++) {
                if (edge1_.solution[a] != nullptr) {
                    delete[] edge1_.solution[a];
                    edge1_.solution[a] = nullptr;
                }
            }
        } else {
            for (cl_long a = 0; a < edge1.numSol / bagSizeEdge1; a++) {
                if (edge1.solution[a] != nullptr) {
                    delete[] edge1.solution[a];
                    edge1.solution[a] = nullptr;
                }
            }
        }
        if (node.variables.size() != edge2.variables.size()) {
            for (cl_long a = 0; a < edge2_.numSol / bagSizeEdge2; a++) {
                if (edge2_.solution[a] != nullptr) {
                    delete[] edge2_.solution[a];
                    edge2_.solution[a] = nullptr;
                }
            }
        } else {
            for (cl_long a = 0; a < edge2.numSol / bagSizeEdge2; a++) {
                if (edge2.solution[a] != nullptr) {
                    delete[] edge2.solution[a];
                    edge2.solution[a] = nullptr;
                }
            }

        }
        if (this->getStats) {
            cl_long numSpath = 0;
            this->numHoldPaths.push_back(numHpath);
            for (cl_long a = 0; a < numIterations; a++) {
                if (node.solution[a] != nullptr) {
                    for (cl_long b = 0; b < bagSizeNode; b++) {
                        if (node.solution[a][b] > 0) {
                            numSpath++;
                        }
                    }
                }
            }
            this->numSolPaths.push_back(numSpath);
        }
#ifdef DEBUG
        std::cout << "Join:\n";
        GPUSATUtils::printSol(node.numSol, node.variables, node.solution, formula, bagSizeNode);
#endif
    }
}