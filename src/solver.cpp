#include <algorithm>
#include <cmath>
#include <iostream>
#include <gpusautils.h>
#include <solver.h>

namespace gpusat {
    void Solver::solveProblem(treedecType &decomp, satformulaType &formula, bagType &node, bagType &pnode) {
        if (isSat > 0) {
            if (node.edges.size() == 0) {
                bagType cNode;
                myHashTable t;
                std::vector<myTableElement> elems(1);
                elems[0].id = 0;
                elems[0].count = 1.0;
                t.elements = elems;
                t.maxId = 1;
                t.minId = 0;
                t.numSolutions = 1;
                cNode.solution.resize(1);
                cNode.solution[0] = t;
                solveIntroduceForget(formula, pnode, node, cNode, true);
            } else if (node.edges.size() == 1) {
                solveProblem(decomp, formula, *node.edges[0], node);
                if (isSat == 1) {
                    bagType &cnode = *node.edges[0];
                    solveIntroduceForget(formula, pnode, node, cnode, false);
                }
            } else if (node.edges.size() > 1) {
                bagType &edge1 = *node.edges[0];
                solveProblem(decomp, formula, edge1, node);
                if (isSat <= 0) {
                    return;
                }
                if (isSat == 1) {
                    bagType tmp, edge2_, edge1_;

                    for (cl_long i = 1; i < node.edges.size(); i++) {
                        bagType &edge2 = *node.edges[i];
                        solveProblem(decomp, formula, edge2, node);
                        if (isSat <= 0) {
                            return;
                        }

                        std::vector<cl_long> vt(static_cast<unsigned long long int>(edge1.variables.size() + edge2.variables.size()));
                        auto itt = std::set_union(edge1.variables.begin(), edge1.variables.end(), edge2.variables.begin(), edge2.variables.end(), vt.begin());
                        vt.resize(static_cast<unsigned long long int>(itt - vt.begin()));
                        tmp.variables = vt;
                        solveJoin(tmp, edge1, edge2, formula);
                        edge1 = tmp;

                        if (i == node.edges.size() - 1) {
                            solveIntroduceForget(formula, pnode, node, tmp, false);
                        }
                    }
                }
            }
        }
    }

    void Solver::resizeMap(myHashTable &table) {
        myHashTable t;
        t.numSolutions = 0;
        t.elements.resize(table.numSolutions * 2);
        t.minId = table.minId;
        t.maxId = table.maxId;
        combineMap(t, table);
    }

    void Solver::combineMap(myHashTable &to, myHashTable &from) {
        if (from.elements.size() > 0) {
            cl::Kernel kernel_resize = cl::Kernel(program, "resize");
            cl::Buffer buf_sols_old(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(myTableElement) * from.elements.size(), &from.elements[0]);
            kernel_resize.setArg(0, buf_sols_old);
            cl::Buffer buf_sols_new(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(myTableElement) * to.elements.size(), &to.elements[0]);
            kernel_resize.setArg(1, buf_sols_new);
            kernel_resize.setArg(2, to.elements.size());
            cl::Buffer buf_num_sol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &to.numSolutions);
            kernel_resize.setArg(3, buf_num_sol);

            cl_long error1 = 0, error2 = 0;
            error1 = queue.enqueueNDRangeKernel(kernel_resize, cl::NDRange(static_cast<size_t>(0)), cl::NDRange(static_cast<size_t>(from.elements.size())));
            error2 = queue.finish();
            if (error1 != 0 || error2 != 0) {
                std::cerr << "Resize - OpenCL error: " << (error1 != 0 ? error1 : error2) << "\n";
                exit(1);
            }
            queue.enqueueReadBuffer(buf_sols_new, CL_TRUE, 0, sizeof(myTableElement) * to.elements.size(), &to.elements[0]);
            queue.enqueueReadBuffer(buf_num_sol, CL_TRUE, 0, sizeof(cl_long), &to.numSolutions);
        }
        to.minId = std::min(from.minId, to.minId);
        to.maxId = std::max(from.maxId, to.maxId);
        from.elements.clear();
        from = to;
    }

    void Solver_Primal::solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula) {
        cl_long bagSizeNode = static_cast<cl_long>(pow(2, std::min(maxWidth, (cl_long) node.variables.size())));

        node.solution.resize(ceil((1 << (node.variables.size())) / bagSizeNode));
        this->numJoin++;
        cl::Kernel kernel = cl::Kernel(program, "solveJoin");
        cl::Buffer bufSolVars;
        if (node.variables.size() > 0) {
            bufSolVars = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * node.variables.size(), &node.variables[0]);
            kernel.setArg(3, bufSolVars);
        } else {
            kernel.setArg(3, NULL);
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
        cl::Buffer bufWeights;
        if (formula.variableWeights != nullptr) {
            bufWeights = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * formula.numWeights, formula.variableWeights);
            kernel.setArg(16, bufWeights);
        } else {
            kernel.setArg(16, NULL);
        }

        for (cl_long a = 0; a < node.solution.size(); a++) {
            node.solution[a].elements.resize(bagSizeNode);
            for (cl_long i = 0; i < bagSizeNode; i++) {
                node.solution[a].elements[i].count = -1.0;
                node.solution[a].elements[i].id = i + a * bagSizeNode;
            }
            if (a > 0) {
                node.solution[a].minId = node.solution[a - 1].maxId;
                node.solution[a].maxId = node.solution[a].minId + bagSizeNode;
            } else {
                node.solution[a].minId = 0;
                node.solution[a].maxId = bagSizeNode;
            }
            cl::Buffer bufSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(myTableElement) * node.solution[a].elements.size(), &node.solution[a].elements[0]);
            kernel.setArg(0, bufSol);
            cl_long bagsolutions = 0;
            cl::Buffer bufsolBag(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &node.solution[a].numSolutions);
            kernel.setArg(17, bufsolBag);
            kernel.setArg(13, node.solution[a].minId);
            kernel.setArg(18, node.solution[a].elements.size());

            for (cl_long b = 0; b < std::max(edge1.solution.size(), edge2.solution.size()); b++) {
                cl::Buffer bufSol1;
                if (b < edge1.solution.size() && edge1.solution[b].numSolutions != 0) {
                    bufSol1 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(myTableElement) * edge1.solution[b].elements.size(), &edge1.solution[b].elements[0]);
                    kernel.setArg(1, bufSol1);
                } else {
                    kernel.setArg(1, NULL);
                }

                cl_long minIdE1 = -1;
                cl_long maxIdE1 = -1;
                cl_long startIDE1 = 0;
                cl_long tableSizeE1 = 0;
                if (b < edge1.solution.size()) {
                    minIdE1 = edge1.solution[b].minId;
                    maxIdE1 = edge1.solution[b].maxId;
                    startIDE1 = edge1.solution[b].minId;
                    tableSizeE1 = edge1.solution[b].elements.size();
                }
                kernel.setArg(9, minIdE1);
                kernel.setArg(10, maxIdE1);
                kernel.setArg(14, startIDE1);
                kernel.setArg(19, tableSizeE1);

                cl::Buffer bufSol2;
                if (b < edge2.solution.size() && edge2.solution[b].numSolutions != 0) {
                    bufSol2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(myTableElement) * edge2.solution[b].elements.size(), &edge2.solution[b].elements[0]);
                    kernel.setArg(2, bufSol2);
                } else {
                    kernel.setArg(2, NULL);
                }

                cl_long minIdE2 = -1;
                cl_long maxIdE2 = -1;
                cl_long startIDE2 = 0;
                cl_long tableSizeE2 = 0;
                if (b < edge2.solution.size()) {
                    minIdE2 = edge2.solution[b].minId;
                    maxIdE2 = edge2.solution[b].maxId;
                    startIDE2 = edge2.solution[b].minId;
                    tableSizeE2 = edge2.solution[b].elements.size();
                }
                kernel.setArg(11, minIdE2);
                kernel.setArg(12, maxIdE2);
                kernel.setArg(15, startIDE2);
                kernel.setArg(20, tableSizeE2);


                cl_long error1 = 0, error2 = 0;
                error1 = queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(node.solution[a].minId)), cl::NDRange(static_cast<size_t>(bagSizeNode)));
                error2 = queue.finish();
                if (error1 != 0 || error2 != 0) {
                    std::cerr << "Join - OpenCL error: " << (error1 != 0 ? error1 : error2) << "\n";
                    exit(1);
                }
            }
            queue.enqueueReadBuffer(bufsolBag, CL_TRUE, 0, sizeof(cl_long), &bagsolutions);
            node.solution[a].numSolutions += bagsolutions;
            if (node.solution[a].numSolutions == 0) {
                node.solution[a].elements.clear();
            } else {
                queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(myTableElement) * node.solution[a].elements.size(), &node.solution[a].elements[0]);
                this->isSat = 1;
                if (a > 0 && node.solution[a].numSolutions + node.solution[a - 1].numSolutions < bagSizeNode) {
                    this->combineMap(node.solution[a], node.solution[a - 1]);
                    node.solution.erase(node.solution.begin() + a);
                    a--;
                }
                if (node.solution[a].elements.size() > node.solution[a].numSolutions * 2) {
                    this->resizeMap(node.solution[a]);
                }
            }
        }
        //std::cout << "Join\n";
        //GPUSATUtils::printSol(node);
        for (cl_long a = 0; a < edge1.solution.size(); a++) {
            edge1.solution[a].elements.clear();
        }
        for (cl_long a = 0; a < edge2.solution.size(); a++) {
            edge2.solution[a].elements.clear();
        }
    }

    void Solver_Primal::solveIntroduceForget(satformulaType &formula, bagType &pnode, bagType &node, bagType &cnode, bool leaf) {
        isSat = 0;
        std::vector<cl_long> fVars;
        std::set_intersection(node.variables.begin(), node.variables.end(), pnode.variables.begin(), pnode.variables.end(), std::back_inserter(fVars));
        std::vector<cl_long> iVars = node.variables;
        std::vector<cl_long> eVars = cnode.variables;

        node.variables = fVars;

        cl_long bagSizeForget = static_cast<cl_long>(pow(2, std::min((cl_long) fVars.size(), maxWidth)));

        node.solution.resize(ceil((1 << (node.variables.size())) / bagSizeForget));
        this->numIntroduceForget++;

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

        cl::Kernel kernel = cl::Kernel(program, "solveIntroduceForget");

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
            bufWeights = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * formula.numWeights, formula.variableWeights);
            kernel.setArg(17, bufWeights);
        } else {
            kernel.setArg(17, NULL);
        }
        for (cl_long a = 0; a < node.solution.size(); a++) {
            node.solution[a].elements.resize(bagSizeForget);
            if (a > 0) {
                node.solution[a].minId = node.solution[a - 1].maxId;
                node.solution[a].maxId = node.solution[a].minId + bagSizeForget;
            } else {
                node.solution[a].minId = 0;
                node.solution[a].maxId = bagSizeForget;
            }
            cl::Buffer buf_solsF(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(myTableElement) * node.solution[a].elements.size(), &node.solution[a].elements[0]);
            kernel.setArg(0, buf_solsF);
            cl::Buffer buf_varsF;
            if (fVars.size() > 0) {
                buf_varsF = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * fVars.size(), &fVars[0]);
                kernel.setArg(1, buf_varsF);
            } else {
                kernel.setArg(1, NULL);
            }
            kernel.setArg(9, node.solution[a].minId);

            cl_long bagsolutions = 0;
            cl::Buffer bufsolBag(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &bagsolutions);
            kernel.setArg(11, bufsolBag);
            for (cl_long b = 0; b < cnode.solution.size(); b++) {
                if (cnode.solution[b].elements.size() == 0) {
                    continue;
                }
                cl::Buffer buf_solsE(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(myTableElement) * cnode.solution[b].elements.size(), &cnode.solution[b].elements[0]);
                if (!leaf) {
                    kernel.setArg(2, buf_solsE);
                } else {
                    kernel.setArg(2, NULL);
                }
                kernel.setArg(7, cnode.solution[b].minId);
                kernel.setArg(8, cnode.solution[b].maxId);
                kernel.setArg(10, cnode.solution[b].minId);
                kernel.setArg(18, node.solution[a].elements.size());
                kernel.setArg(19, cnode.solution[b].elements.size());

                cl_long error1 = 0, error2 = 0;
                error1 = queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(node.solution[a].minId)), cl::NDRange(static_cast<size_t>(bagSizeForget)));
                error2 = queue.finish();
                if (error1 != 0 || error2 != 0) {
                    std::cerr << "Introduce Forget - OpenCL error: " << (error1 != 0 ? error1 : error2) << "\n";
                    exit(1);
                }
            }
            queue.enqueueReadBuffer(bufsolBag, CL_TRUE, 0, sizeof(cl_long), &bagsolutions);
            node.solution[a].numSolutions += bagsolutions;
            if (node.solution[a].numSolutions == 0) {
                node.solution[a].elements.clear();
            } else {
                queue.enqueueReadBuffer(buf_solsF, CL_TRUE, 0, sizeof(myTableElement) * node.solution[a].elements.size(), &node.solution[a].elements[0]);
                this->isSat = 1;
                if (a > 0 && node.solution[a].numSolutions + node.solution[a - 1].numSolutions < bagSizeForget) {
                    this->combineMap(node.solution[a], node.solution[a - 1]);
                    node.solution.erase(node.solution.begin() + a);
                    a--;
                }
                if (node.solution[a].elements.size() > node.solution[a].numSolutions * 2) {
                    this->resizeMap(node.solution[a]);
                }
            }
        }
        //std::cout << "IF\n";
        //GPUSATUtils::printSol(node);
        for (cl_long a = 0; a < cnode.solution.size(); a++) {
            cnode.solution.clear();
        }
    }
/*
    void Solver_Incidence::solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula) {
        bagType edge1_;
        this->numJoin++;
        edge1_.variables = node.variables;
        edge1_.numSol = node.numSol;
        if (node.variables.size() != edge1.variables.size()) {
            solveIntroduceForget(formula, node, edge1_, edge1, false);
        } else {
            edge1_.solution = edge1.solution;
        }

        bagType edge2_;
        edge2_.variables = node.variables;
        edge2_.numSol = node.numSol;
        if (node.variables.size() != edge2.variables.size()) {
            solveIntroduceForget(formula, node, edge2_, edge2, false);
        } else {
            edge2_.solution = edge2.solution;
        }

        auto bagSizeEdge1 = static_cast<cl_long>(pow(2, std::min(maxWidth, (cl_long) edge1_.variables.size())));
        auto bagSizeEdge2 = static_cast<cl_long>(pow(2, std::min(maxWidth, (cl_long) edge2_.variables.size())));
        auto bagSizeNode = static_cast<cl_long>(pow(2, std::min(maxWidth, (cl_long) node.variables.size())));
        node.solution = new cl_double *[(cl_long) ceil(node.numSol / bagSizeNode)];

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
        cl::Kernel kernel = cl::Kernel(program, "solveJoin");
        //number of clauses - 10
        cl_long numClauses = nClausesVector.size() - 1;
        kernel.setArg(10, numClauses);
        //weights - 11
        cl::Buffer bufWeights;
        if (formula.variableWeights != nullptr) {
            bufWeights = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * formula.numWeights, formula.variableWeights);
            kernel.setArg(11, bufWeights);
        } else {
            kernel.setArg(11, NULL);
        }
        //node variables - 12
        cl::Buffer bufVariables = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (nVars.size()), &nVars[0]);
        kernel.setArg(12, bufVariables);
        for (int a = 0; a < numIterations; a++) {
            cl_int solutions = 0;
            node.solution[a] = new cl_double[bagSizeNode]();
            //node solutions - 0
            cl::Buffer bufSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * (bagSizeNode), node.solution[a]);
            kernel.setArg(0, bufSol);
            //start id node - 7
            cl_long startIdNode = a * bagSizeNode;
            kernel.setArg(7, startIdNode);
            cl_int bagsolutions = 0;
            cl::Buffer bufsolBag(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &bagsolutions);
            kernel.setArg(13, bufsolBag);
            for (int b = 0; b < numIterations; b++) {
                if (edge1_.solution[b] == nullptr) {
                    continue;
                }
                //edge1 solutions - 1
                cl_long es1 = (b < pow(2, std::max((cl_long) 0, (cl_long) edge1_.variables.size() - maxWidth))) || b == 0 ? bagSizeEdge1 : 0;
                cl::Buffer bufSol1;
                if (es1 > 0) {
                    bufSol1 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * (es1), edge1_.solution[b]);
                    kernel.setArg(1, bufSol1);
                } else {
                    kernel.setArg(1, NULL);
                }
                //min id edge1 - 3
                cl_long minId1 = ((b + 1) <= pow(2, std::max((cl_long) 0, (cl_long) edge1_.variables.size() - maxWidth)) || b == 0 ? bagSizeEdge1 * b : -1);
                kernel.setArg(3, minId1);
                //max id edge1 - 4
                cl_long maxId1 = ((b + 1) <= pow(2, std::max((cl_long) 0, (cl_long) edge1_.variables.size() - maxWidth)) || b == 0 ? bagSizeEdge1 * (b + 1) : -1);
                kernel.setArg(4, maxId1);
                //start id edge1 - 8
                cl_long startIdEdge1 = b * bagSizeEdge1;
                kernel.setArg(8, startIdEdge1);
                for (int c = 0; c < numIterations; c++) {
                    if (edge2_.solution[c] == nullptr) {
                        continue;
                    }
                    //edge2 solutions - 2
                    cl_long es2 = (c < pow(2, std::max((cl_long) 0, (cl_long) edge2_.variables.size() - maxWidth))) || c == 0 ? bagSizeEdge2 : 0;
                    cl::Buffer bufSol2;
                    if (es2 > 0) {
                        bufSol2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * (es2), edge2_.solution[c]);
                        kernel.setArg(2, bufSol2);
                    } else {

                        kernel.setArg(2, NULL);
                    }
                    //min id edge2 - 5
                    cl_long minId2 = ((c + 1) <= pow(2, std::max((cl_long) 0, (cl_long) edge2_.variables.size() - maxWidth)) || c == 0 ? bagSizeEdge2 * c : -1);
                    kernel.setArg(5, minId2);
                    //max id edge2 - 6
                    cl_long maxId2 = ((c + 1) <= pow(2, std::max((cl_long) 0, (cl_long) edge2_.variables.size() - maxWidth)) || c == 0 ? bagSizeEdge2 * (c + 1) : -1);
                    kernel.setArg(6, maxId2);
                    //start id edge2 - 9
                    cl_long startIdEdge2 = c * bagSizeEdge2;
                    kernel.setArg(9, startIdEdge2);

                    int error1 = 0, error2 = 0;
                    error1 = queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(startIdNode)), cl::NDRange(static_cast<size_t>(bagSizeNode)));
                    error2 = queue.finish();
                    if (error1 != 0 || error2 != 0) {
                        std::cerr << "Join - OpenCL error: " << (error1 != 0 ? error1 : error2) << "\n";
                        exit(1);
                    }
                }
            }
            queue.enqueueReadBuffer(bufsolBag, CL_TRUE, 0, sizeof(cl_int), &bagsolutions);
            if (bagsolutions > 0) {
                solutions = 1;
            }
            if (solutions == 0) {
                delete[] node.solution[a];
                node.solution[a] = nullptr;
            } else {
                queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_double) * (bagSizeNode), node.solution[a]);
                this->isSat = 1;
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
    }

    void Solver_Incidence::solveIntroduceForget(satformulaType &formula, bagType &pnode, bagType &node, bagType &cnode, bool leaf) {
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
        node.solution = new cl_double *[(cl_long) ceil(node.numSol / bagSizeNode)];
        this->numIntroduceForget++;

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

        cl_long numSubIterations = static_cast<cl_long>(pow(2, std::max((cl_long) 0, (cl_long) eNode.size() - maxWidth)));
        cl::Kernel kernel = cl::Kernel(program, "solveIntroduceForget");
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
        cl::Buffer bufIVars = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * iVars.size(), &iVars[0]);
        kernel.setArg(15, bufIVars);
        kernel.setArg(16, iVars.size() - 1);
        cl::Buffer bufIClauses = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * iClauses.size(), &iClauses[0]);
        kernel.setArg(17, bufIClauses);
        kernel.setArg(18, iClauses.size() - 1);
        for (int a = 0; a < numIterations; a++) {
            cl_int solutions = 0;
            node.solution[a] = new cl_double[bagSizeNode]();
            //node solutions - 0
            cl::Buffer bufNodeSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * (bagSizeNode), node.solution[a]);
            kernel.setArg(0, bufNodeSol);
            cl_long startIdNode = a * bagSizeNode;
            //start id node - 10
            kernel.setArg(10, startIdNode);
            //clauses - 2
            cl::Buffer bufClauses = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (clauseVector.size()),
                                               &clauseVector[0]);
            kernel.setArg(19, bufClauses);
            //clauses array length - 3
            cl_long clausesLength = clauseVector.size();
            kernel.setArg(20, clausesLength);

            cl::Buffer bufWeights;
            if (formula.variableWeights != nullptr) {
                bufWeights = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * formula.numWeights, formula.variableWeights);
                kernel.setArg(21, bufWeights);
            } else {
                kernel.setArg(21, NULL);
            }

            cl_int bagsolutions = 0;
            cl::Buffer bufsolBag(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &bagsolutions);
            kernel.setArg(14, bufsolBag);
            for (int b = 0; b < numSubIterations; b++) {
                if (cnode.solution[b] == nullptr) {
                    continue;
                }
                cl::Buffer bufEdgeSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * (bagSizeEdge), cnode.solution[b]);
                //edge solutions - 1
                if (!leaf) {
                    kernel.setArg(1, bufEdgeSol);
                } else {
                    kernel.setArg(1, NULL);
                }
                //start id edge - 11
                cl_long startIdEdge = b * bagSizeEdge;
                kernel.setArg(11, startIdEdge);
                //min id edge - 12
                cl_long minId = b * bagSizeEdge;
                kernel.setArg(12, minId);
                //max id edge - 13
                cl_long maxId = (b + 1) * bagSizeEdge;
                kernel.setArg(13, maxId);

                int error1 = 0, error2 = 0;
                error1 = queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(startIdNode)), cl::NDRange(static_cast<size_t>(bagSizeNode)));
                error2 = queue.finish();
                if (error1 != 0 || error2 != 0) {
                    std::cerr << "Intoduce Forget - OpenCL error: " << (error1 != 0 ? error1 : error2) << "\n";
                    exit(1);
                }
            }
            queue.enqueueReadBuffer(bufsolBag, CL_TRUE, 0, sizeof(cl_int), &bagsolutions);
            if (bagsolutions != 0) {
                solutions = 1;
                isSat = 1;
            }
            if (solutions == 0) {
                delete[] node.solution[a];
                node.solution[a] = nullptr;
            } else {
                queue.enqueueReadBuffer(bufNodeSol, CL_TRUE, 0, sizeof(cl_double) * (bagSizeNode), node.solution[a]);
                this->isSat = 1;
            }
        }
        for (cl_long a = 0; a < cnode.numSol / bagSizeEdge; a++) {
            if (cnode.solution[a] != nullptr) {
                delete[] cnode.solution[a];
                cnode.solution[a] = nullptr;
            }
        }
    }

    void Solver_Dual::solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula) {
        dual::sJVars params;
        cl_long bagSizeEdge1 = static_cast<cl_long>(pow(2, std::min(maxWidth, (cl_long) edge1.variables.size())));
        cl_long bagSizeEdge2 = static_cast<cl_long>(pow(2, std::min(maxWidth, (cl_long) edge2.variables.size())));
        cl_long bagSizeNode = static_cast<cl_long>(pow(2, std::min(maxWidth, (cl_long) node.variables.size())));
        node.solution = new cl_double *[(cl_long) ceil(node.numSol / bagSizeNode)];
        cl_long numIterations = (cl_long) ceil(node.numSol / bagSizeNode);
        cl_long numIterationsEdge1 = (cl_long) ceil(edge1.numSol / bagSizeEdge1);
        cl_long numIterationsEdge2 = (cl_long) ceil(edge2.numSol / bagSizeEdge2);
        this->numJoin++;
        cl::Kernel kernel = cl::Kernel(program, "solveJoin");

        std::set<cl_long> nodeVariablesSet;
        std::vector<cl_long> clauseVars;
        std::vector<cl_long> numVarsC;
        for (cl_long i = 0; i < node.variables.size(); i++) {
            for (int j = 0; j < formula.clauses[node.variables[i] - 1].size(); ++j) {
                nodeVariablesSet.insert(abs(formula.clauses[node.variables[i] - 1][j]));
            }
            clauseVars.insert(clauseVars.end(), formula.clauses[node.variables[i] - 1].begin(), formula.clauses[node.variables[i] - 1].end());
            numVarsC.push_back(formula.clauses[node.variables[i] - 1].size());
        }
        std::vector<cl_long> nodeVariables;
        nodeVariables.insert(nodeVariables.begin(), nodeVariablesSet.begin(), nodeVariablesSet.end());
        std::sort(nodeVariables.begin(), nodeVariables.end());

        std::set<cl_long> edge1VariablesSet;
        std::vector<cl_long> clauseVarsEdge1;
        std::vector<cl_long> numVarsCE1;
        std::vector<cl_long> varsE1;
        for (cl_long i = 0; i < edge1.variables.size(); i++) {
            for (int j = 0; j < formula.clauses[edge1.variables[i] - 1].size(); ++j) {
                edge1VariablesSet.insert(abs(formula.clauses[edge1.variables[i] - 1][j]));
            }
            clauseVarsEdge1.insert(clauseVarsEdge1.end(), formula.clauses[edge1.variables[i] - 1].begin(), formula.clauses[edge1.variables[i] - 1].end());
            numVarsCE1.push_back(formula.clauses[edge1.variables[i] - 1].size());
        }
        varsE1.insert(varsE1.begin(), edge1VariablesSet.begin(), edge1VariablesSet.end());
        std::sort(varsE1.begin(), varsE1.end());

        std::set<cl_long> edge2VariablesSet;
        std::vector<cl_long> clauseVarsEdge2;
        std::vector<cl_long> numVarsCE2;
        std::vector<cl_long> varsE2;
        for (cl_long i = 0; i < edge2.variables.size(); i++) {
            for (int j = 0; j < formula.clauses[edge2.variables[i] - 1].size(); ++j) {
                edge2VariablesSet.insert(abs(formula.clauses[edge2.variables[i] - 1][j]));
            }
            clauseVarsEdge2.insert(clauseVarsEdge2.end(), formula.clauses[edge2.variables[i] - 1].begin(), formula.clauses[edge2.variables[i] - 1].end());
            numVarsCE2.push_back(formula.clauses[edge2.variables[i] - 1].size());
        }
        varsE2.insert(varsE2.begin(), edge2VariablesSet.begin(), edge2VariablesSet.end());
        std::sort(varsE2.begin(), varsE2.end());

        params.numVE2 = varsE2.size();
        params.numVE1 = varsE1.size();
        cl::Buffer bufvarsE1s;
        if (varsE1.size() > 0) {
            bufvarsE1s = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * varsE1.size(), &varsE1[0]);
            kernel.setArg(11, bufvarsE1s);
        } else {
            kernel.setArg(11, NULL);
        }
        cl::Buffer bufvarsE2;
        if (varsE2.size() > 0) {
            bufvarsE2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * varsE2.size(), &varsE2[0]);
            kernel.setArg(12, bufvarsE2);
        } else {
            kernel.setArg(12, NULL);
        }
        cl::Buffer bufnumVarsCE1;
        if (numVarsCE1.size() > 0) {
            bufnumVarsCE1 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * numVarsCE1.size(), &numVarsCE1[0]);
            kernel.setArg(13, bufnumVarsCE1);
        } else {
            kernel.setArg(13, NULL);
        }
        cl::Buffer bufnumVarsCE2;
        if (numVarsCE2.size() > 0) {
            bufnumVarsCE2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * numVarsCE2.size(), &numVarsCE2[0]);
            kernel.setArg(14, bufnumVarsCE2);
        } else {
            kernel.setArg(14, NULL);
        }
        cl::Buffer bufclauseVarsE1;
        if (clauseVarsEdge1.size() > 0) {
            bufclauseVarsE1 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * clauseVarsEdge1.size(), &clauseVarsEdge1[0]);
            kernel.setArg(15, bufclauseVarsE1);
        } else {
            kernel.setArg(15, NULL);
        }
        cl::Buffer bufclauseVarsE2;
        if (node.variables.size() > 0) {
            bufclauseVarsE2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * clauseVarsEdge2.size(), &clauseVarsEdge2[0]);
            kernel.setArg(16, bufclauseVarsE2);
        } else {
            kernel.setArg(16, NULL);
        }

        cl::Buffer bufclauseVars;
        if (clauseVars.size() > 0) {
            bufclauseVars = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * clauseVars.size(), &clauseVars[0]);
            kernel.setArg(5, bufclauseVars);
        } else {
            kernel.setArg(5, NULL);
        }
        cl::Buffer bufnumVarsC;
        if (numVarsC.size() > 0) {
            bufnumVarsC = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * numVarsC.size(), &numVarsC[0]);
            kernel.setArg(9, bufnumVarsC);
        } else {
            kernel.setArg(9, NULL);
        }
        cl::Buffer bufvariables;
        if (nodeVariables.size() > 0) {
            bufvariables = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * nodeVariables.size(), &nodeVariables[0]);
            kernel.setArg(10, bufvariables);
        } else {
            kernel.setArg(10, NULL);
        }
        params.numV = nodeVariables.size();
        //clauseIds
        cl::Buffer bufClauseIds;
        if (node.variables.size() > 0) {
            bufClauseIds = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * node.variables.size(), &node.variables[0]);
            kernel.setArg(6, bufClauseIds);
        } else {
            kernel.setArg(6, NULL);
        }
        //clauseIdsE1
        cl::Buffer bufClauseIdsE1;
        if (edge1.variables.size() > 0) {
            bufClauseIdsE1 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * edge1.variables.size(), &edge1.variables[0]);
            kernel.setArg(7, bufClauseIdsE1);
        } else {
            kernel.setArg(7, NULL);
        }
        //clauseIdsE2
        cl::Buffer bufClauseIdsE2;
        if (edge2.variables.size() > 0) {
            bufClauseIdsE2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * edge2.variables.size(), &edge2.variables[0]);
            kernel.setArg(8, bufClauseIdsE2);
        } else {
            kernel.setArg(8, NULL);
        }
        params.numC = node.variables.size();
        params.numCE1 = edge1.variables.size();
        params.numCE2 = edge2.variables.size();

        for (int a = 0; a < numIterations; a++) {
            cl_int solutions = 0;
            node.solution[a] = new cl_double[bagSizeNode]();
            for (cl_long i = 0; i < bagSizeNode; i++) {
                node.solution[a][i] = 1.0;
            }
            cl_long startIdNode = a * bagSizeNode;
            cl::Buffer bufSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * (bagSizeNode), node.solution[a]);
            kernel.setArg(1, bufSol);
            cl_int bagsolutions = 0;
            cl::Buffer bufsolBag(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &bagsolutions);
            kernel.setArg(4, bufsolBag);

            for (int b = 0; b < std::max(numIterationsEdge1, numIterationsEdge2); b++) {
                params.startIDEdge1 = b * bagSizeEdge1;
                params.startIDEdge2 = b * bagSizeEdge2;
                cl::Buffer bufSol1;
                if (b < numIterationsEdge1 && edge1.solution[b] != nullptr) {
                    bufSol1 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * (bagSizeEdge1), edge1.solution[b]);
                    kernel.setArg(2, bufSol1);
                } else {
                    kernel.setArg(2, NULL);
                }
                cl::Buffer bufSol2;
                if (b < numIterationsEdge2 && edge2.solution[b] != nullptr) {
                    bufSol2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * (bagSizeEdge2), edge2.solution[b]);
                    kernel.setArg(3, bufSol2);
                } else {
                    kernel.setArg(3, NULL);
                }


                params.minId1 = ((b + 1) <= pow(2, std::max((cl_long) 0, (cl_long) edge1.variables.size() - maxWidth)) || b == 0 ? bagSizeEdge1 * b : -1);
                params.maxId1 = ((b + 1) <= pow(2, std::max((cl_long) 0, (cl_long) edge1.variables.size() - maxWidth)) || b == 0 ? bagSizeEdge1 * (b + 1) : -1);
                params.minId2 = ((b + 1) <= pow(2, std::max((cl_long) 0, (cl_long) edge2.variables.size() - maxWidth)) || b == 0 ? bagSizeEdge2 * b : -1);
                params.maxId2 = ((b + 1) <= pow(2, std::max((cl_long) 0, (cl_long) edge2.variables.size() - maxWidth)) || b == 0 ? bagSizeEdge2 * (b + 1) : -1);
                params.startIDNode = startIdNode;

                kernel.setArg(0, params);

                int error1 = 0, error2 = 0;
                error1 = queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(startIdNode)), cl::NDRange(static_cast<size_t>(bagSizeNode)));
                error2 = queue.finish();
                if (error1 != 0 || error2 != 0) {
                    std::cerr << "Join - OpenCL error: " << (error1 != 0 ? error1 : error2) << "\n";
                    exit(1);
                }
            }
            queue.enqueueReadBuffer(bufsolBag, CL_TRUE, 0, sizeof(cl_int), &bagsolutions);
            solutions += bagsolutions;
            if (solutions == 0) {
                delete[] node.solution[a];
                node.solution[a] = nullptr;
            } else {
                queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_double) * (bagSizeNode), node.solution[a]);
                this->isSat = 1;
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
    }

    void Solver_Dual::solveIntroduceForget(satformulaType &formula, bagType &pnode, bagType &node, bagType &cnode, bool leaf) {
        isSat = 0;
        dual::sIFVars params;
        std::vector<cl_long> clauseIdsF;
        std::set_intersection(node.variables.begin(), node.variables.end(), pnode.variables.begin(), pnode.variables.end(), std::back_inserter(clauseIdsF));
        std::vector<cl_long> clauseIdsI = node.variables;
        std::vector<cl_long> clauseIdsE = cnode.variables;
        params.numCE = clauseIdsE.size();
        params.numCI = clauseIdsI.size();
        params.numCF = clauseIdsF.size();
        params.combinations = static_cast<cl_long>(pow(2, params.numCI - clauseIdsF.size()));


        std::set<cl_long> varsFSet;
        std::vector<cl_long> clauseVarsF;
        std::vector<cl_long> numVarsF;
        for (cl_long i = 0; i < params.numCF; i++) {
            for (int j = 0; j < formula.clauses[clauseIdsF[i] - 1].size(); ++j) {
                varsFSet.insert(abs(formula.clauses[clauseIdsF[i] - 1][j]));
            }
            clauseVarsF.insert(clauseVarsF.end(), formula.clauses[clauseIdsF[i] - 1].begin(), formula.clauses[clauseIdsF[i] - 1].end());
            numVarsF.push_back(formula.clauses[clauseIdsF[i] - 1].size());
        }
        std::vector<cl_long> varsF;
        varsF.insert(varsF.begin(), varsFSet.begin(), varsFSet.end());
        std::sort(varsF.begin(), varsF.end());

        params.numVF = varsF.size();

        std::set<cl_long> varsESet;
        std::vector<cl_long> clauseVarsE;
        std::vector<cl_long> numVarsE;
        for (cl_long i = 0; i < params.numCE; i++) {
            for (int j = 0; j < formula.clauses[clauseIdsE[i] - 1].size(); ++j) {
                varsESet.insert(abs(formula.clauses[clauseIdsE[i] - 1][j]));
            }
            clauseVarsE.insert(clauseVarsE.end(), formula.clauses[clauseIdsE[i] - 1].begin(), formula.clauses[clauseIdsE[i] - 1].end());
            numVarsE.push_back(formula.clauses[clauseIdsE[i] - 1].size());
        }
        std::vector<cl_long> varsE;
        varsE.insert(varsE.begin(), varsESet.begin(), varsESet.end());
        std::sort(varsE.begin(), varsE.end());

        params.numVE = varsE.size();

        std::set<cl_long> varsISet;
        std::vector<cl_long> clauseVarsI;
        std::vector<cl_long> numVarsI;
        for (cl_long i = 0; i < params.numCI; i++) {
            for (int j = 0; j < formula.clauses[node.variables[i] - 1].size(); ++j) {
                varsISet.insert(abs(formula.clauses[clauseIdsI[i] - 1][j]));
            }
            clauseVarsI.insert(clauseVarsI.end(), formula.clauses[clauseIdsI[i] - 1].begin(), formula.clauses[clauseIdsI[i] - 1].end());
            numVarsI.push_back(formula.clauses[clauseIdsI[i] - 1].size());
        }
        std::vector<cl_long> varsI;
        varsI.insert(varsI.begin(), varsISet.begin(), varsISet.end());
        std::sort(varsI.begin(), varsI.end());

        params.numVI = varsI.size();

        node.variables = clauseIdsF;
        node.numSol = pow(2, node.variables.size());

        cl_long bagSizeEdge = static_cast<cl_long>(pow(2, std::min((cl_long) clauseIdsE.size(), maxWidth)));
        cl_long bagSizeForget = static_cast<cl_long>(pow(2, std::min((cl_long) clauseIdsF.size(), maxWidth)));

        cl_long numIterations = (cl_long) ceil(node.numSol / bagSizeForget);
        cl_long numSubIterations = 1;
        if (!leaf) { numSubIterations = (cl_long) ceil(cnode.numSol / bagSizeEdge); }

        node.solution = new cl_double *[(cl_long) ceil(node.numSol / bagSizeForget)];
        this->numIntroduceForget++;

        cl::Kernel kernel = cl::Kernel(program, "solveIntroduceForget");

        cl::Buffer buf_clauseIdsE;
        if (clauseIdsE.size() > 0) {
            buf_clauseIdsE = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * clauseIdsE.size(), &clauseIdsE[0]);
            kernel.setArg(5, buf_clauseIdsE);
        } else {
            kernel.setArg(5, NULL);
        }

        //variables introduce
        cl::Buffer buf_clauseIdsI;
        if (clauseIdsI.size() > 0) {
            buf_clauseIdsI = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * clauseIdsI.size(), &clauseIdsI[0]);
            kernel.setArg(4, buf_clauseIdsI);
        } else {
            kernel.setArg(4, NULL);
        }

        cl::Buffer buf_clauseIdsF;
        if (clauseIdsF.size() > 0) {
            buf_clauseIdsF = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (clauseIdsF.size()), &clauseIdsF[0]);
            kernel.setArg(3, buf_clauseIdsF);
        } else {
            kernel.setArg(3, NULL);
        }

        cl::Buffer buf_varsE;
        if (varsE.size() > 0) {
            buf_varsE = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (varsE.size()), &varsE[0]);
            kernel.setArg(7, buf_varsE);
        } else {
            kernel.setArg(7, NULL);
        }

        cl::Buffer buf_varsF;
        if (varsF.size() > 0) {
            buf_varsF = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (varsF.size()), &varsF[0]);
            kernel.setArg(6, buf_varsF);
        } else {
            kernel.setArg(6, NULL);
        }

        cl::Buffer buf_varsI;
        if (varsI.size() > 0) {
            buf_varsI = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (varsI.size()), &varsI[0]);
            kernel.setArg(8, buf_varsI);
        } else {
            kernel.setArg(8, NULL);
        }

        cl::Buffer buf_numVarsE;
        if (numVarsE.size() > 0) {
            buf_numVarsE = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (numVarsE.size()), &numVarsE[0]);
            kernel.setArg(10, buf_numVarsE);
        } else {
            kernel.setArg(10, NULL);
        }

        cl::Buffer buf_numVarsI;
        if (numVarsI.size() > 0) {
            buf_numVarsI = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (numVarsI.size()), &numVarsI[0]);
            kernel.setArg(11, buf_numVarsI);
        } else {
            kernel.setArg(11, NULL);
        }

        cl::Buffer buf_clauseVarsI;
        if (clauseVarsI.size() > 0) {
            buf_clauseVarsI = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (clauseVarsI.size()), &clauseVarsI[0]);
            kernel.setArg(12, buf_clauseVarsI);
        } else {
            kernel.setArg(12, NULL);
        }

        cl::Buffer buf_clauseVarsE;
        if (clauseVarsE.size() > 0) {
            buf_clauseVarsE = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (clauseVarsE.size()), &clauseVarsE[0]);
            kernel.setArg(13, buf_clauseVarsE);
        } else {
            kernel.setArg(13, NULL);
        }

        for (int a = 0; a < numIterations; a++) {
            cl_int solutions = 0;
            node.solution[a] = new cl_double[bagSizeForget]();
            params.startIDF = a * bagSizeForget;
            cl::Buffer buf_solsF(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * (bagSizeForget), node.solution[a]);
            kernel.setArg(1, buf_solsF);

            cl_int bagsolutions = 0;
            cl::Buffer bufsolBag(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &bagsolutions);
            kernel.setArg(9, bufsolBag);
            for (int b = 0; b < numSubIterations; b++) {
                if (cnode.solution[b] == nullptr) {
                    continue;
                }
                cl::Buffer buf_solsE(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * (bagSizeEdge), cnode.solution[b]);
                if (!leaf) {
                    kernel.setArg(2, buf_solsE);
                } else {
                    kernel.setArg(2, NULL);
                }
                params.minIdE = b * bagSizeEdge;
                params.maxIdE = (b + 1) * bagSizeEdge;
                params.startIDE = b * bagSizeEdge;

                int error1 = 0, error2 = 0;
                kernel.setArg(0, params);
                error1 = queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(params.startIDF)), cl::NDRange(static_cast<size_t>(bagSizeForget)));
                error2 = queue.finish();
                if (error1 != 0 || error2 != 0) {
                    std::cerr << "Intoduce Forget - OpenCL error: " << (error1 != 0 ? error1 : error2) << "\n";
                    exit(1);
                }
            }
            queue.enqueueReadBuffer(bufsolBag, CL_TRUE, 0, sizeof(cl_int), &bagsolutions);
            solutions += bagsolutions;
            if (solutions == 0) {
                delete[] node.solution[a];
                node.solution[a] = nullptr;
            } else {
                queue.enqueueReadBuffer(buf_solsF, CL_TRUE, 0, sizeof(cl_double) * (bagSizeForget), node.solution[a]);
                this->isSat = 1;
            }
        }
        for (cl_long a = 0; a < cnode.numSol / bagSizeEdge; a++) {
            if (cnode.solution[a] != nullptr) {
                delete[] cnode.solution[a];
                cnode.solution[a] = nullptr;
            }
        }
    }
    */
}