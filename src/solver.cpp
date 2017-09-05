#include <solver.h>
#include <algorithm>
#include <math.h>

namespace gpusat {
    void Solver::solveProblem(treedecType &decomp, satformulaType &formula, bagType &node) {

        if (isSat > 0) {
            node.solution = new solType[node.numSol]();
            if (node.numEdges == 0) {
                solveLeaf(formula, node);
            } else if (node.numEdges == 1) {
                cl_long edge = node.edges[0] - 1;
                solveProblem(decomp, formula, decomp.bags[edge]);
                if (isSat <= 0) {
                    return;
                }
                bagType &next = decomp.bags[node.edges[0] - 1];
                solveForgIntroduce(formula, node, next);
                free(next.solution);

            } else if (node.numEdges > 1) {
                cl_long edge = node.edges[0] - 1;
                solveProblem(decomp, formula, decomp.bags[edge]);
                if (isSat <= 0) {
                    return;
                }
                bagType tmp, edge2_, edge1_, &edge1 = decomp.bags[node.edges[0] - 1];
                std::vector<cl_long> v1(edge1.numVars + node.numVars);
                std::vector<cl_long>::iterator it1 = std::set_intersection(node.variables,
                                                                           node.variables + node.numVars,
                                                                           edge1.variables,
                                                                           edge1.variables + edge1.numVars,
                                                                           v1.begin());
                v1.resize(it1 - v1.begin());
                edge1_.variables = &v1[0];
                edge1_.numVars = v1.size();
                edge1_.numSol = pow(2, edge1_.numVars);
                edge1_.solution = new solType[edge1_.numSol]();
                solveForget(edge1_, edge1);
                free(edge1.solution);

                for (int i = 1; i < node.numEdges; i++) {
                    cl_long edge = node.edges[i] - 1;
                    solveProblem(decomp, formula, decomp.bags[edge]);
                    if (isSat <= 0) {
                        return;
                    }
                    bagType &edge2 = decomp.bags[node.edges[i] - 1];
                    std::vector<cl_long> v2(edge2.numVars + node.numVars);
                    std::vector<cl_long>::iterator it2 = std::set_intersection(node.variables,
                                                                               node.variables + node.numVars,
                                                                               edge2.variables,
                                                                               edge2.variables + edge2.numVars,
                                                                               v2.begin());
                    v2.resize(it2 - v2.begin());
                    edge2_.variables = &v2[0];
                    edge2_.numVars = v2.size();
                    edge2_.numSol = pow(2, edge2_.numVars);
                    edge2_.solution = new solType[edge2_.numSol]();
                    solveForget(edge2_, edge2);
                    free(edge2.solution);

                    std::vector<cl_long> vt(edge1_.numVars + edge2_.numVars);
                    std::vector<cl_long>::iterator itt = std::set_union(edge1_.variables,
                                                                        edge1_.variables + edge1_.numVars,
                                                                        edge2_.variables,
                                                                        edge2_.variables + edge2_.numVars,
                                                                        vt.begin());
                    vt.resize(itt - vt.begin());
                    tmp.variables = new cl_long[vt.size()];
                    memcpy(tmp.variables, &vt[0], vt.size() * sizeof(cl_long));
                    tmp.numVars = vt.size();
                    tmp.numSol = pow(2, tmp.numVars);
                    tmp.solution = new solType[tmp.numSol]();
                    solveJoin(tmp, edge1_, edge2_);

                    free(edge1_.solution);
                    free(edge2_.solution);
                    edge1_ = tmp;

                    if (i == node.numEdges - 1) {
                        solveIntroduce(formula, node, tmp);
                    }
                }
            }
        }
    }

    void Solver::solveForgIntroduce(satformulaType &formula, bagType &node, bagType &next) {

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
            edge.solution = new solType[edge.numSol]();

            solveForget(edge, next);
            solveIntroduce(formula, node, edge);
            free(edge.solution);
        }
    }

    void Solver::solveIntroduce(satformulaType &formula, bagType &node, bagType &edge) {
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
                              sizeof(solType) * (edge.numSol),
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
        queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(solType) * (node.numSol), node.solution);
        queue.enqueueReadBuffer(bufSAT, CL_TRUE, 0, sizeof(cl_long), &isSat);
    }

    void Solver::solveLeaf(satformulaType &formula, bagType &node) {
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
        queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(solType) * (node.numSol), node.solution);
        queue.enqueueReadBuffer(bufSAT, CL_TRUE, 0, sizeof(cl_long), &isSat);
    }

    void Solver::solveForget(bagType &node, bagType &edge) {
        kernel = cl::Kernel(program, "solveForget");
        cl::Buffer bufSol(context,
                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                          sizeof(solType) * (node.numSol),
                          node.solution);
        cl::Buffer bufNextSol(context,
                              CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(solType) * (edge.numSol),
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
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(node.numSol));
        queue.finish();
        queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(solType) * (node.numSol), node.solution);
    }

    void Solver::solveJoin(bagType &node, bagType &edge1, bagType &edge2) {
        kernel = cl::Kernel(program, "solveJoin");
        cl::Buffer bufSol(context,
                          CL_MEM_READ_WRITE,
                          sizeof(cl_long) * (node.numSol));
        cl::Buffer bufSol1(context,
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(solType) * (edge1.numSol),
                           edge1.solution);
        cl::Buffer bufSol2(context,
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           sizeof(solType) * (edge2.numSol),
                           edge2.solution);
        cl::Buffer bufSolVars;
        if (node.numVars > 0) {
            bufSolVars = cl::Buffer(context,
                                    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl_long) * node.numVars,
                                    node.variables);
            kernel.setArg(3, bufSolVars);
        } else {
            kernel.setArg(3, NULL);
        }
        cl::Buffer bufSolVars1;
        if (edge1.numVars > 0) {
            bufSolVars1 = cl::Buffer(context,
                                     CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     sizeof(cl_long) * edge1.numVars,
                                     edge1.variables);
            kernel.setArg(4, bufSolVars1);
        } else {
            kernel.setArg(4, NULL);
        }
        cl::Buffer bufSolVars2;
        if (edge2.numVars > 0) {
            bufSolVars2 = cl::Buffer(context,
                                     CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     sizeof(cl_long) * edge2.numVars,
                                     edge2.variables);
            kernel.setArg(5, bufSolVars2);
        } else {
            kernel.setArg(5, NULL);
        }
        kernel.setArg(0, bufSol);
        kernel.setArg(1, bufSol1);
        kernel.setArg(2, bufSol2);
        kernel.setArg(6, node.numVars);
        kernel.setArg(7, edge1.numVars);
        kernel.setArg(8, edge2.numVars);
        queue.enqueueNDRangeKernel(kernel, cl::NDRange(0), cl::NDRange(node.numSol));
        queue.finish();
        queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(solType) * (node.numSol), node.solution);
    }
}
