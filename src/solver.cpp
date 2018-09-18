#include <algorithm>
#include <cmath>
#include <iostream>
#include <gpusautils.h>
#include <solver.h>

namespace gpusat {
    void Solver::solveProblem(treedecType &decomp, satformulaType &formula, bagType &node, bagType &pnode) {
        if (isSat > 0) {
            if (node.edges.empty()) {
                bagType cNode;
                cNode.solution = new myHashTable[1];
                cNode.solution[0].elements = new myTableElement[1];
                cNode.solution[0].elements[0].count = 1.0;
                cNode.solution[0].elements[0].id = 0;
                cNode.solution[0].maxId = 1;
                cNode.solution[0].minId = 0;
                cNode.solution[0].numSolutions = 1;
                cNode.solution[0].size = 1;
                cNode.bags = 1;
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
        t.size = (static_cast<unsigned long>(table.numSolutions * 2));
        t.elements = new myTableElement[(static_cast<unsigned long>(table.numSolutions * 2))];
        t.minId = table.minId;
        t.maxId = table.maxId;
        combineMap(t, table);
    }

    void Solver::combineMap(myHashTable &to, myHashTable &from) {
        if (from.size > 0) {
            cl::Kernel kernel_resize = cl::Kernel(program, "resize");
            cl::Buffer buf_sols_old(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(myTableElement) * from.size, from.elements);
            kernel_resize.setArg(0, buf_sols_old);
            cl::Buffer buf_sols_new(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(myTableElement) * to.size, to.elements);
            kernel_resize.setArg(1, buf_sols_new);
            kernel_resize.setArg(2, to.size);
            cl::Buffer buf_num_sol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &to.numSolutions);
            kernel_resize.setArg(3, buf_num_sol);

            cl_long error1 = 0, error2 = 0;
            error1 = queue.enqueueNDRangeKernel(kernel_resize, cl::NDRange(static_cast<size_t>(0)), cl::NDRange(static_cast<size_t>(from.size)));
            error2 = queue.finish();
            if (error1 != 0 || error2 != 0) {
                std::cerr << "Resize - OpenCL error: " << (error1 != 0 ? error1 : error2) << "\n";
                exit(1);
            }
            queue.enqueueReadBuffer(buf_sols_new, CL_TRUE, 0, sizeof(myTableElement) * to.size, to.elements);
            queue.enqueueReadBuffer(buf_num_sol, CL_TRUE, 0, sizeof(cl_long), &to.numSolutions);
        }
        to.minId = std::min(from.minId, to.minId);
        to.maxId = std::max(from.maxId, to.maxId);
        delete[] from.elements;
        from = to;
    }

    void Solver_Primal::solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula) {
        cl_long bagSizeNode = 1l << std::min(maxWidth, (cl_long) node.variables.size());

        node.solution = new myHashTable[(static_cast<unsigned long>(ceil((1l << (node.variables.size())) / bagSizeNode)))];
        node.bags = ((static_cast<unsigned long>(ceil((1l << (node.variables.size())) / bagSizeNode))));
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

        for (cl_long a = 0; a < node.bags; a++) {
            node.solution[a].elements = new myTableElement[(bagSizeNode)];
            node.solution[a].size = (bagSizeNode);
            for (cl_long i = 0; i < bagSizeNode; i++) {
                node.solution[a].elements[i].count = -1.0;
                node.solution[a].elements[i].id = i + a * bagSizeNode;
            }
            node.solution[a].minId = a * bagSizeNode;
            node.solution[a].maxId = a * bagSizeNode + bagSizeNode;
            cl::Buffer bufSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(myTableElement) * node.solution[a].size, node.solution[a].elements);
            kernel.setArg(0, bufSol);
            cl_long bagsolutions = 0;
            cl::Buffer bufsolBag(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &node.solution[a].numSolutions);
            kernel.setArg(17, bufsolBag);
            kernel.setArg(13, node.solution[a].minId);
            kernel.setArg(18, node.solution[a].size);

            for (cl_long b = 0; b < std::max(edge1.bags, edge2.bags); b++) {
                cl::Buffer bufSol1;
                if (b < edge1.bags && edge1.solution[b].numSolutions != 0) {
                    bufSol1 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(myTableElement) * edge1.solution[b].size, edge1.solution[b].elements);
                    kernel.setArg(1, bufSol1);
                } else {
                    kernel.setArg(1, NULL);
                }

                kernel.setArg(9, (b < edge1.bags) ? edge1.solution[b].minId : -1);
                kernel.setArg(10, (b < edge1.bags) ? edge1.solution[b].maxId : -1);
                kernel.setArg(14, (b < edge1.bags) ? edge1.solution[b].minId : 0);
                kernel.setArg(19, (b < edge1.bags) ? edge1.solution[b].size : 0);

                cl::Buffer bufSol2;
                if (b < edge2.bags && edge2.solution[b].numSolutions != 0) {
                    bufSol2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(myTableElement) * edge2.solution[b].size, edge2.solution[b].elements);
                    kernel.setArg(2, bufSol2);
                } else {
                    kernel.setArg(2, NULL);
                }

                kernel.setArg(11, (b < edge2.bags) ? edge2.solution[b].minId : -1);
                kernel.setArg(12, (b < edge2.bags) ? edge2.solution[b].maxId : -1);
                kernel.setArg(15, (b < edge2.bags) ? edge2.solution[b].minId : 0);
                kernel.setArg(20, (b < edge2.bags) ? edge2.solution[b].size : 0);

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
                delete[] node.solution[a].elements;
                node.solution[a].elements = NULL;
                node.solution[a].size = 0;
            } else {
                queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(myTableElement) * node.solution[a].size, node.solution[a].elements);
                this->isSat = 1;
                /*if (a > 0 && ((node.solution[a].numSolutions + node.solution[a - 1].numSolutions) * 2 < bagSizeNode)) {
                    this->combineMap(node.solution[a-1], node.solution[a]);

                    a--;
                    node.bags--;
                    for(cl_long e = a;e<node.bags;e++){
                        node.solution[e]=node.solution[e+1];
                    }
                }*/

                if (node.solution[a].size > (node.solution[a].numSolutions * 2)) {
                    this->resizeMap(node.solution[a]);
                }
            }
        }
        cl_long tableSize = 0;
        for (cl_long i = 0; i < node.bags; i++) {
            tableSize += node.solution[i].size;
        }
        this->maxTableSize = std::max(this->maxTableSize, tableSize);
        for (cl_long a = 0; a < edge1.bags; a++) {
            if (edge1.solution[a].elements != NULL) {
                edge1.solution[a].size = 0;
                delete[] edge1.solution[a].elements;
            }
        }
        for (cl_long a = 0; a < edge2.bags; a++) {
            if (edge2.solution[a].elements != NULL) {
                edge2.solution[a].size = 0;
                delete[] edge2.solution[a].elements;
            }
        }
        std::cout << "Join " << numJoin << "\n";
        GPUSATUtils::printSol(node);
        std::cout.flush();
    }

    void Solver_Primal::solveIntroduceForget(satformulaType &formula, bagType &pnode, bagType &node, bagType &cnode, bool leaf) {
        isSat = 0;
        std::vector<cl_long> fVars;
        std::set_intersection(node.variables.begin(), node.variables.end(), pnode.variables.begin(), pnode.variables.end(), std::back_inserter(fVars));
        std::vector<cl_long> iVars = node.variables;
        std::vector<cl_long> eVars = cnode.variables;

        node.variables = fVars;

        cl_long bagSizeForget = static_cast<cl_long>(pow(2, std::min((cl_long) fVars.size(), maxWidth)));

        node.solution = new myHashTable[(static_cast<unsigned long>(ceil((1l << (node.variables.size())) / bagSizeForget)))];
        node.bags = ((static_cast<unsigned long>(ceil((1l << (node.variables.size())) / bagSizeForget))));
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

        for (cl_long a = 0; a < node.bags; a++) {
            node.solution[a].numSolutions = 0;
            node.solution[a].elements = new myTableElement[(bagSizeForget)];
            node.solution[a].size = bagSizeForget;
            node.solution[a].minId = a * bagSizeForget;
            node.solution[a].maxId = a * bagSizeForget + bagSizeForget;

            cl::Buffer buf_solsF(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(myTableElement) * node.solution[a].size, node.solution[a].elements);
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
            for (cl_long b = 0; b < cnode.bags; b++) {
                if (cnode.solution[b].size == 0) {
                    continue;
                }
                cl::Buffer buf_solsE(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(myTableElement) * cnode.solution[b].size, cnode.solution[b].elements);
                if (!leaf) {
                    kernel.setArg(2, buf_solsE);
                } else {
                    kernel.setArg(2, NULL);
                }
                kernel.setArg(7, cnode.solution[b].minId);
                kernel.setArg(8, cnode.solution[b].maxId);
                kernel.setArg(10, cnode.solution[b].minId);

                kernel.setArg(18, node.solution[a].size);
                kernel.setArg(19, cnode.solution[b].size);

                cl_long error1 = 0, error2 = 0;
                error1 = queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(node.solution[a].minId)), cl::NDRange(static_cast<size_t>(node.solution[a].size)));
                error2 = queue.finish();
                if (error1 != 0 || error2 != 0) {
                    std::cerr << "Introduce Forget - OpenCL error: " << (error1 != 0 ? error1 : error2) << "\n";
                    exit(1);
                }
            }
            queue.enqueueReadBuffer(bufsolBag, CL_TRUE, 0, sizeof(cl_long), &bagsolutions);
            node.solution[a].numSolutions += bagsolutions;
            if (node.solution[a].numSolutions == 0) {
                delete[] node.solution[a].elements;
                node.solution[a].elements = NULL;
                node.solution[a].size = 0;
            } else {
                queue.enqueueReadBuffer(buf_solsF, CL_TRUE, 0, sizeof(myTableElement) * node.solution[a].size, node.solution[a].elements);
                this->isSat = 1;
                /*if (a > 0 && ((node.solution[a].numSolutions + node.solution[a - 1].numSolutions) * 2 < bagSizeForget)) {
                    this->combineMap(node.solution[a-1], node.solution[a]);

                    a--;
                    node.bags--;
                    for(cl_long e = a;e<node.bags;e++){
                        node.solution[e]=node.solution[e+1];
                    }
                }*/

                if (node.solution[a].size > node.solution[a].numSolutions * 2) {
                    this->resizeMap(node.solution[a]);
                }
            }
        }
        cl_long tableSize = 0;
        for (cl_long i = 0; i < node.bags; i++) {
            tableSize += node.solution[i].size;
        }
        this->maxTableSize = std::max(this->maxTableSize, tableSize);
        for (cl_long a = 0; a < cnode.bags; a++) {
            if (cnode.solution[a].elements != NULL) {
                cnode.solution[a].size = (0);
                delete[]cnode.solution[a].elements;
            }
        }
        std::cout << "IF " << numIntroduceForget << "\n";
        GPUSATUtils::printSol(node);
        std::cout.flush();
    }
}