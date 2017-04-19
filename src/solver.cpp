#include <gpusatutils.h>
#include <math.h>
#include <solver.h>

void
solveProblem(treedecType decomp, satformulaType formula, bagType node, cl::Context my_context, cl::Kernel my_kernel,
             cl::Program my_program, cl::CommandQueue my_queue) {
    for (int i = 0; i < node.nume; i++) {
        cl_long edge = node.edges[i] - 1;
        solveProblem(decomp, formula, decomp.bags[edge], cl::Context(), cl::Kernel(), cl::Program(),
                     cl::CommandQueue());
    }

    cl::Buffer bufSol = cl::Buffer(my_context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(cl_long) * (node.numSol * (node.numv + 1)),
                                   node.solution);
    cl::Buffer bufVertices = cl::Buffer(my_context,
                                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        sizeof(cl_long) * (node.numv),
                                        node.vertices);
    if (node.nume > 1) {
        //join node
        long size = 0;
        for (int i = 0; i < node.nume; i++) {
            size += decomp.bags[node.edges[i] - 1].numSol * (decomp.bags[node.edges[i] - 1].numv + 1);
        }
        cl_long *solutions = new cl_long[size];
        long s = 0;
        for (int i = 0; i < node.nume; i++) {
            std::copy(decomp.bags[node.edges[i] - 1].solution,
                      decomp.bags[node.edges[i] - 1].solution +
                      decomp.bags[node.edges[i] - 1].numSol * (decomp.bags[node.edges[i] - 1].numv + 1), solutions + s);
            s += decomp.bags[node.edges[i] - 1].numSol * (decomp.bags[node.edges[i] - 1].numv + 1);
        }
        cl::Buffer bufSolOther = cl::Buffer(my_context,
                                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                            sizeof(cl_long) * (s),
                                            solutions);
        my_kernel = cl::Kernel(my_program, "solveJoin");
        my_kernel.setArg(0, bufSol);
        my_kernel.setArg(1, node.numv);
        my_kernel.setArg(2, node.nume);
        my_kernel.setArg(3, bufVertices);
        my_kernel.setArg(4, bufSolOther);
        my_kernel.setArg(5, node.numSol);
/*
        solveJoin(node.solution, node.numv, node.nume, node.vertices, solutions, node.numSol);
*/
    } else if (node.nume == 1 && decomp.bags[node.edges[0] - 1].numv > node.numv) {
        //forget node
        cl::Buffer bufSolNext = cl::Buffer(my_context,
                                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                           sizeof(cl_long) * (decomp.bags[node.edges[0] - 1].numSol *
                                                              (decomp.bags[node.edges[0] - 1].numv + 1)),
                                           decomp.bags[node.edges[0] - 1].solution);
        my_kernel = cl::Kernel(my_program, "solveForget");
        my_kernel.setArg(0, bufSol);
        my_kernel.setArg(1, node.numv);
        my_kernel.setArg(2, bufSolNext);
        my_kernel.setArg(3, decomp.bags[node.edges[0] - 1].numv);
        my_kernel.setArg(4, decomp.bags[node.edges[0] - 1].numSol);
        my_kernel.setArg(5, bufVertices);
/*
        solveForget(node.solution, node.numv, decomp.bags[node.edges[0] - 1].solution,
                    decomp.bags[node.edges[0] - 1].numv, decomp.bags[node.edges[0] - 1].numSol, node.vertices);
*/
    } else {
        if (node.nume == 0) {
            //leaf node
            cl::Buffer bufClauses = cl::Buffer(my_context,
                                               CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                               sizeof(cl_long) * (formula.totalNumVar),
                                               formula.clauses);
            cl::Buffer bufNumVarsC = cl::Buffer(my_context,
                                                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                sizeof(cl_long) * (formula.numclauses),
                                                formula.numVarsC);
            my_kernel = cl::Kernel(my_program, "solveLeaf");
            my_kernel.setArg(0, bufClauses);
            my_kernel.setArg(1, bufNumVarsC);
            my_kernel.setArg(2, formula.numclauses);
            my_kernel.setArg(3, bufSol);
            my_kernel.setArg(4, node.numv);
            my_kernel.setArg(5, bufVertices);
/*
            solveLeaf(formula.clauses, formula.numVarsC, formula.numclauses, node.solution, node.numv,
                      node.vertices);
*/
        } else if (node.nume == 1 && decomp.bags[node.edges[0] - 1].numv < node.numv) {
            //introduce node
            cl::Buffer bufClauses = cl::Buffer(my_context,
                                               CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                               sizeof(cl_long) * (formula.totalNumVar),
                                               formula.clauses);
            cl::Buffer bufNumVarsC = cl::Buffer(my_context,
                                                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                sizeof(cl_long) * (formula.numclauses),
                                                formula.numVarsC);
            cl::Buffer bufSolNext = cl::Buffer(my_context,
                                               CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                               sizeof(cl_long) * (decomp.bags[node.edges[0] - 1].numSol *
                                                                  (decomp.bags[node.edges[0] - 1].numv + 1)),
                                               decomp.bags[node.edges[0] - 1].solution);
            my_kernel = cl::Kernel(my_program, "solveIntroduce");
            my_kernel.setArg(0, bufClauses);
            my_kernel.setArg(1, bufNumVarsC);
            my_kernel.setArg(2, formula.numclauses);
            my_kernel.setArg(3, bufSol);
            my_kernel.setArg(4, node.numv);
            my_kernel.setArg(5, bufSolNext);
            my_kernel.setArg(6, decomp.bags[node.edges[0] - 1].numv);
            my_kernel.setArg(7, decomp.bags[node.edges[0] - 1].numSol);
            my_kernel.setArg(8, bufVertices);
/*
            solveIntroduce(formula.clauses, formula.numVarsC, formula.numclauses, node.solution, node.numv,
                           decomp.bags[node.edges[0] - 1].solution,
                           decomp.bags[node.edges[0] - 1].numv, decomp.bags[node.edges[0] - 1].numSol,
                           node.vertices);
*/
        }
    }
    my_queue.enqueueNDRangeKernel(my_kernel, cl::NDRange(), cl::NDRange((size_t) pow(2, node.numv)),
                                  cl::NDRange(CL_DEVICE_MAX_WORK_GROUP_SIZE));
    my_queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_long) * (node.numSol * (node.numv + 1)), node.solution);
}
