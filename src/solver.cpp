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
                cNode.solution = new treeType[1];
                cNode.solution[0].elements = static_cast<cl_long *>(calloc(sizeof(cl_long), 1));
                double val = 1.0;
                cNode.solution[0].elements[0] = *reinterpret_cast <cl_long *>(&val);
                cNode.solution[0].maxId = 1;
                cNode.solution[0].minId = 0;
                cNode.solution[0].numSolutions = 0;
                cNode.solution[0].size = 1;
                cNode.bags = 1;
                cNode.maxSize = 1;
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

    void Solver::cleanTree(treeType &table, cl_long size, cl_long numVars) {
        treeType t;
        t.numSolutions = 0;
        t.size = size + numVars;
        t.elements = static_cast<cl_long *>(calloc(sizeof(cl_long), size + numVars));
        t.minId = table.minId;
        t.maxId = table.maxId;
        if (table.size > 0) {
            cl::Kernel kernel_resize = cl::Kernel(program, "resize");

            kernel_resize.setArg(0, numVars);

            cl::Buffer buf_sols_new(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * t.size, t.elements);
            kernel_resize.setArg(1, buf_sols_new);

            cl::Buffer buf_sols_old(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * table.size, table.elements);
            kernel_resize.setArg(2, buf_sols_old);

            cl::Buffer buf_num_sol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &t.numSolutions);
            kernel_resize.setArg(3, buf_num_sol);

            kernel_resize.setArg(4, table.minId);

            cl_long error1 = 0, error2 = 0;
            error1 = queue.enqueueNDRangeKernel(kernel_resize, cl::NDRange(static_cast<size_t>(0)), cl::NDRange(static_cast<size_t>(table.size)));
            error2 = queue.finish();
            if (error1 != 0 || error2 != 0) {
                std::cerr << "\nResize 1 - OpenCL error: " << (error1 != 0 ? error1 : error2) << "\n";
                exit(1);
            }
            queue.enqueueReadBuffer(buf_sols_new, CL_TRUE, 0, sizeof(cl_long) * t.size, t.elements);
            queue.enqueueReadBuffer(buf_num_sol, CL_TRUE, 0, sizeof(cl_long), &t.numSolutions);
        }
        t.minId = std::min(table.minId, t.minId);
        t.maxId = std::max(table.maxId, t.maxId);
        free(table.elements);
        table = t;
    }

    void Solver::combineTree(treeType &t, treeType &table, cl_long numVars) {
        if (table.size > 0) {
            cl::Kernel kernel_resize = cl::Kernel(program, "resize_");

            kernel_resize.setArg(0, numVars);

            cl::Buffer buf_sols_new(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * t.size, t.elements);
            kernel_resize.setArg(1, buf_sols_new);

            cl::Buffer buf_sols_old(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * table.size, table.elements);
            kernel_resize.setArg(2, buf_sols_old);

            cl::Buffer buf_num_sol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &t.numSolutions);
            kernel_resize.setArg(3, buf_num_sol);

            kernel_resize.setArg(4, table.minId);

            cl_long error1 = 0, error2 = 0;
            error1 = queue.enqueueNDRangeKernel(kernel_resize, cl::NDRange(static_cast<size_t>(0)), cl::NDRange(static_cast<size_t>(table.maxId - table.minId)));
            error2 = queue.finish();
            if (error1 != 0 || error2 != 0) {
                std::cerr << "\nResize 2 - OpenCL error: " << (error1 != 0 ? error1 : error2) << "\n";
                exit(1);
            }
            queue.enqueueReadBuffer(buf_sols_new, CL_TRUE, 0, sizeof(cl_long) * t.size, t.elements);
            queue.enqueueReadBuffer(buf_num_sol, CL_TRUE, 0, sizeof(cl_long), &t.numSolutions);
        }
        t.minId = std::min(table.minId, t.minId);
        t.maxId = std::max(table.maxId, t.maxId);
        free(table.elements);
    }

    void Solver_Primal::solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula) {
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

        cl_long usedMemory = sizeof(cl_long) * node.variables.size() + sizeof(cl_long) * edge1.variables.size() + sizeof(cl_long) * edge2.variables.size() + sizeof(cl_double) * formula.numWeights;

        cl_long s = sizeof(cl_long);
        cl_long bagSizeNode = std::min((cl_long) std::min((memorySize - usedMemory - edge1.maxSize * s - edge2.maxSize * s) / s / 2, (memorySize - usedMemory) / 2 / 3 / s), 1l << node.variables.size());
        bagSizeNode = 1l << (cl_long) std::floor(std::log2(bagSizeNode));

        cl_long maxSize = std::ceil((1l << (node.variables.size())) * 1.0 / bagSizeNode);
        node.solution = new treeType[maxSize];
        node.bags = maxSize;
        for (cl_long a = 0, run = 0; a < node.bags; a++, run++) {
            node.solution[a].numSolutions = 0;
            node.solution[a].minId = run * bagSizeNode;
            node.solution[a].maxId = std::min(run * bagSizeNode + bagSizeNode, 1l << (node.variables.size()));
            node.solution[a].size = (node.solution[a].maxId - node.solution[a].minId) * 2 + node.variables.size();
            node.solution[a].elements = static_cast<cl_long *>(calloc(sizeof(cl_long), node.solution[a].size));

            for (cl_long i = 0; i < bagSizeNode; i++) {
                double val = -1.0;
                node.solution[a].elements[i] = *reinterpret_cast <cl_long *>(&val);
            }
            cl::Buffer bufSol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * node.solution[a].size, node.solution[a].elements);
            kernel.setArg(0, bufSol);
            cl::Buffer bufsolBag(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &node.solution[a].numSolutions);
            kernel.setArg(17, bufsolBag);
            kernel.setArg(13, node.solution[a].minId);

            for (cl_long b = 0; b < std::max(edge1.bags, edge2.bags); b++) {
                cl::Buffer bufSol1;
                if (b < edge1.bags && edge1.solution[b].elements != NULL) {
                    bufSol1 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * edge1.solution[b].size, edge1.solution[b].elements);
                    kernel.setArg(1, bufSol1);
                } else {
                    kernel.setArg(1, NULL);
                }

                kernel.setArg(9, (b < edge1.bags) ? edge1.solution[b].minId : -1);
                kernel.setArg(10, (b < edge1.bags) ? edge1.solution[b].maxId : -1);
                kernel.setArg(14, (b < edge1.bags) ? edge1.solution[b].minId : 0);

                cl::Buffer bufSol2;
                if (b < edge2.bags && edge2.solution[b].elements != NULL) {
                    bufSol2 = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * edge2.solution[b].size, edge2.solution[b].elements);
                    kernel.setArg(2, bufSol2);
                } else {
                    kernel.setArg(2, NULL);
                }

                kernel.setArg(11, (b < edge2.bags) ? edge2.solution[b].minId : -1);
                kernel.setArg(12, (b < edge2.bags) ? edge2.solution[b].maxId : -1);
                kernel.setArg(15, (b < edge2.bags) ? edge2.solution[b].minId : 0);

                cl_long error1 = 0, error2 = 0;
                error1 = queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(node.solution[a].minId)), cl::NDRange(static_cast<size_t>(node.solution[a].maxId - node.solution[a].minId)));
                error2 = queue.finish();
                if (error1 != 0 || error2 != 0) {
                    std::cerr << "\nJoin - OpenCL error: " << (error1 != 0 ? error1 : error2) << "\n";
                    exit(1);
                }
            }
            queue.enqueueReadBuffer(bufsolBag, CL_TRUE, 0, sizeof(cl_long), &node.solution[a].numSolutions);
            if (node.solution[a].numSolutions == 0) {
                free(node.solution[a].elements);
                node.solution[a].elements = NULL;
            } else {
                queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(long) * node.solution[a].size, node.solution[a].elements);
                this->isSat = 1;
                this->cleanTree(node.solution[a], (bagSizeNode) * 2, node.variables.size());

                if (a > 0 && node.solution[a - 1].elements != NULL && (node.solution[a].numSolutions + node.solution[a - 1].numSolutions + 2) < node.solution[a].size) {
                    combineTree(node.solution[a], node.solution[a - 1], node.variables.size());
                    node.solution[a - 1] = node.solution[a];

                    node.bags--;
                    a--;
                }

                node.solution[a].elements = (cl_long *) realloc(node.solution[a].elements, sizeof(cl_long) * (node.solution[a].numSolutions + 1));
                node.solution[a].size = node.solution[a].numSolutions + 1;
                node.maxSize = std::max(node.maxSize, node.solution[a].size);
            }
        }
        cl_long tableSize = 0;
        for (cl_long i = 0; i < node.bags; i++) {
            tableSize += node.solution[i].size;
        }
        this->maxTableSize = std::max(this->maxTableSize, tableSize);
        for (cl_long a = 0; a < edge1.bags; a++) {
            if (edge1.solution[a].elements != NULL) {
                free(edge1.solution[a].elements);
                edge1.solution[a].elements = NULL;
            }
        }
        for (cl_long a = 0; a < edge2.bags; a++) {
            if (edge2.solution[a].elements != NULL) {
                free(edge2.solution[a].elements);
                edge2.solution[a].elements = NULL;
            }
        }
#ifdef DEBUG
        std::cout << "\nJoin " << numJoin << "\n";
        GPUSATUtils::printSol(node);
        std::cout.flush();
#endif
    }

    void Solver_Primal::solveIntroduceForget(satformulaType &formula, bagType &pnode, bagType &node, bagType &cnode, bool leaf) {
        isSat = 0;
        std::vector<cl_long> fVars;
        std::set_intersection(node.variables.begin(), node.variables.end(), pnode.variables.begin(), pnode.variables.end(), std::back_inserter(fVars));
        std::vector<cl_long> iVars = node.variables;
        std::vector<cl_long> eVars = cnode.variables;

        node.variables = fVars;

        this->numIntroduceForget++;

        std::vector<cl_long> numVarsClause;
        std::vector<cl_long> clauses;
        cl_long numClauses = 0;
        for (cl_long i = 0; i < formula.clauses.size(); i++) {
            std::vector<cl_long> v(formula.clauses[i].size());
            std::vector<cl_long>::iterator it;
            it = std::set_intersection(iVars.begin(), iVars.end(), formula.clauses[i].begin(), formula.clauses[i].end(), v.begin(), compVars);
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

        cl_ulong usedMemory = sizeof(cl_long) * eVars.size() + sizeof(cl_long) * iVars.size() + sizeof(cl_long) * (clauses.size()) + sizeof(cl_long) * (numClauses) + sizeof(cl_double) * formula.numWeights + sizeof(cl_long) * fVars.size();
        cl_long bagSizeForget = 1;
        cl_long s = sizeof(cl_long);
        bagSizeForget = std::min((cl_long) std::min((memorySize - usedMemory - cnode.maxSize * s) / s / 2, (memorySize - usedMemory) / 2 / 3 / s), 1l << node.variables.size());
        bagSizeForget = 1l << (cl_long) std::floor(std::log2(bagSizeForget));

        cl_long maxSize = std::ceil((1l << (node.variables.size())) * 1.0 / bagSizeForget);
        node.solution = new treeType[maxSize];
        node.bags = maxSize;
        for (cl_long a = 0, run = 0; a < node.bags; a++, run++) {
            node.solution[a].numSolutions = 0;
            node.solution[a].minId = run * bagSizeForget;
            node.solution[a].maxId = std::min(run * bagSizeForget + bagSizeForget, 1l << (node.variables.size()));
            node.solution[a].size = (node.solution[a].maxId - node.solution[a].minId) * 2 + node.variables.size();
            node.solution[a].elements = static_cast<cl_long *>(calloc(sizeof(cl_long), node.solution[a].size));

            cl::Buffer buf_solsF(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * node.solution[a].size, node.solution[a].elements);
            kernel.setArg(0, buf_solsF);
            cl::Buffer buf_varsF;
            if (fVars.size() > 0) {
                buf_varsF = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * fVars.size(), &fVars[0]);
                kernel.setArg(1, buf_varsF);
            } else {
                kernel.setArg(1, NULL);
            }
            kernel.setArg(9, node.solution[a].minId);

            cl::Buffer bufsolBag(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &node.solution[a].numSolutions);
            kernel.setArg(11, bufsolBag);
            for (cl_long b = 0; b < cnode.bags; b++) {
                if (cnode.solution[b].elements == NULL) {
                    continue;
                }
                cl::Buffer buf_solsE(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * cnode.solution[b].size, cnode.solution[b].elements);
                if (!leaf) {
                    kernel.setArg(2, buf_solsE);
                } else {
                    kernel.setArg(2, NULL);
                }
                kernel.setArg(7, cnode.solution[b].minId);
                kernel.setArg(8, cnode.solution[b].maxId);
                kernel.setArg(10, cnode.solution[b].minId);

                cl_long error1 = 0, error2 = 0;
                error1 = queue.enqueueNDRangeKernel(kernel, cl::NDRange(static_cast<size_t>(node.solution[a].minId)), cl::NDRange(static_cast<size_t>(node.solution[a].maxId - node.solution[a].minId)));
                error2 = queue.finish();
                if (error1 != 0 || error2 != 0) {
                    std::cerr << "\nIntroduce Forget - OpenCL error: " << (error1 != 0 ? error1 : error2) << "\n";
                    exit(1);
                }
            }
            queue.enqueueReadBuffer(bufsolBag, CL_TRUE, 0, sizeof(cl_long), &node.solution[a].numSolutions);
            if (node.solution[a].numSolutions == 0) {
                free(node.solution[a].elements);
                node.solution[a].elements = NULL;
            } else {
                queue.enqueueReadBuffer(buf_solsF, CL_TRUE, 0, sizeof(long) * node.solution[a].size, node.solution[a].elements);
                this->isSat = 1;

                if (a > 0 && node.solution[a - 1].elements != NULL && (node.solution[a].numSolutions + node.solution[a - 1].numSolutions + 2) < node.solution[a].size) {
                    combineTree(node.solution[a], node.solution[a - 1], node.variables.size());
                    node.solution[a - 1] = node.solution[a];

                    node.bags--;
                    a--;
                }
                node.solution[a].elements = (cl_long *) realloc(node.solution[a].elements, sizeof(cl_long) * (node.solution[a].numSolutions + 1));
                node.solution[a].size = node.solution[a].numSolutions + 1;
                node.maxSize = std::max(node.maxSize, node.solution[a].size);
            }
        }
        cl_long tableSize = 0;
        for (cl_long i = 0; i < node.bags; i++) {
            tableSize += node.solution[i].size;
        }
        this->maxTableSize = std::max(this->maxTableSize, tableSize);
        for (cl_long a = 0; a < cnode.bags; a++) {
            if (cnode.solution[a].elements != NULL) {
                free(cnode.solution[a].elements);
                cnode.solution[a].elements = NULL;
            }
        }
#ifdef DEBUG
        std::cout << "\nIF " << numIntroduceForget << "\n";
        GPUSATUtils::printSol(node);
        std::cout.flush();
#endif
    }
}