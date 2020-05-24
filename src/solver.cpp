#include <algorithm>
#include <cmath>
#include <iostream>
#include <solver.h>
#include <errno.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

extern void introduceForgetWrapper(long *solsF,  long *varsF,  long *solsE, long numVE,  long *varsE, long combinations, long numVF, long minIdE, long maxIdE, long startIDF, long startIDE,  long *sols, long numVI,  long *varsI,  long *clauses,  long *numVarsC, long numclauses,  double *weights,  long *exponent, double value, size_t threads, long id_offset);

namespace gpusat {
    template <typename T>
    class CudaBuffer {
        private:
            size_t buf_size;
        public:
            T* device_mem;
            // creates a buffer with size 0.
            CudaBuffer();
            /// Create an on-device array of T with given length
            CudaBuffer(T* from, size_t length);
            /// Copy a vector to the device.
            /// If the vector is empty, the memory pointer is NULL. 
            CudaBuffer(std::vector<T> vec);
            /// Copy on-device array to `to`
            void read(T* to);
            /// Length of the buffer
            size_t size();

            ~CudaBuffer();
    };

    template <typename T>
    CudaBuffer<T>::CudaBuffer(T* from, size_t length) {
        gpuErrchk(cudaMalloc((void**)&this->device_mem, sizeof(T) * length));
        gpuErrchk(cudaMemcpy(this->device_mem, from, sizeof(T) * length, cudaMemcpyHostToDevice));
        this->buf_size = length;
    }

    template <typename T>
    CudaBuffer<T>::CudaBuffer(std::vector<T> vec) {
        if (vec.size()) {
            gpuErrchk(cudaMalloc((void**)&this->device_mem, sizeof(T) * vec.size()));
            gpuErrchk(cudaMemcpy(this->device_mem, &vec.front(), sizeof(T) * vec.size(), cudaMemcpyHostToDevice));
            this->buf_size = vec.size();
        } else {
            this->buf_size = 0;
            this->device_mem = NULL;
        }
    }

    template <typename T>
    CudaBuffer<T>::CudaBuffer() {
        this->device_mem = NULL;
    }

    template <typename T>
    size_t CudaBuffer<T>::size() {
        return this->buf_size;
    }

    template <typename T>
    void CudaBuffer<T>::read(T* to) {
        gpuErrchk(cudaMemcpy(to, this->device_mem, sizeof(T) * this->size(), cudaMemcpyDeviceToHost));
    }

    template <typename T>
    CudaBuffer<T>::~CudaBuffer() {
        cudaFree(this->device_mem);
    }


    void Solver::solveProblem(treedecType &decomp, satformulaType &formula, bagType &node, bagType &pnode, nodeType lastNode) {

        std::cout << "solve problem. isSAT: " << isSat << " edges:" << node.edges.size() << std::endl;
        if (isSat > 0) {
            if (node.edges.empty()) {
                bagType cNode;
                cNode.solution = new treeType[1];
                if (cNode.solution == NULL || errno == ENOMEM) {
                    std::cerr << "\nOut of Memory\n";
                    exit(0);
                }
                cNode.solution[0].elements = static_cast<cl_long *>(calloc(sizeof(cl_long), 1));
                if (cNode.solution[0].elements == NULL || errno == ENOMEM) {
                    std::cerr << "\nOut of Memory\n";
                    exit(0);
                }
                double val = 1.0;
                cNode.solution[0].elements[0] = *reinterpret_cast <cl_long *>(&val);
                cNode.solution[0].maxId = 1;
                cNode.solution[0].minId = 0;
                cNode.solution[0].numSolutions = 0;
                cNode.solution[0].size = 1;
                cNode.bags = 1;
                cNode.maxSize = 1;
                solveIntroduceForget(formula, pnode, node, cNode, true, lastNode);
            } else if (node.edges.size() == 1) {
                solveProblem(decomp, formula, *node.edges[0], node, INTRODUCEFORGET);
                if (isSat == 1) {
                    bagType &cnode = *node.edges[0];
                    solveIntroduceForget(formula, pnode, node, cnode, false, lastNode);
                }
            } else if (node.edges.size() > 1) {
                bagType &edge1 = *node.edges[0];
                solveProblem(decomp, formula, edge1, node, JOIN);
                if (isSat <= 0) {
                    return;
                }
                if (isSat == 1) {
                    bagType tmp, edge2_, edge1_;

                    for (cl_long i = 1; i < node.edges.size(); i++) {
                        bagType &edge2 = *node.edges[i];
                        solveProblem(decomp, formula, edge2, node, JOIN);
                        if (isSat <= 0) {
                            return;
                        }

                        std::vector<cl_long> vt;
                        std::set_union(edge1.variables.begin(), edge1.variables.end(), edge2.variables.begin(), edge2.variables.end(), back_inserter(vt));
                        tmp.variables = vt;

                        if (i == node.edges.size() - 1) {
                            solveJoin(tmp, edge1, edge2, formula, INTRODUCEFORGET);
                            if (isSat <= 0) {
                                return;
                            }
                            edge1 = tmp;
                            solveIntroduceForget(formula, pnode, node, tmp, false, lastNode);
                        } else {
                            solveJoin(tmp, edge1, edge2, formula, JOIN);
                            edge1 = tmp;
                        }
                    }
                }
            }
        }
    }

    void Solver::cleanTree(treeType &table, cl_long size, cl_long numVars, bagType &node, cl_long nextSize) {
        std::cout << "clean tree" << std::endl;
        treeType t;
        t.numSolutions = 0;
        t.size = size + numVars;
        t.minId = table.minId;
        t.maxId = table.maxId;
        if (table.size > 0) {
            cl::Kernel kernel_resize = cl::Kernel(program, "resize");

            kernel_resize.setArg(0, numVars);

            cl::Buffer buf_sols_old(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * table.size, table.elements);
            kernel_resize.setArg(2, buf_sols_old);

            free(table.elements);

            t.elements = static_cast<cl_long *>(calloc(sizeof(cl_long), size + numVars * 3));
            if (t.elements == NULL || errno == ENOMEM) {
                std::cerr << "\nOut of Memory\n";
                exit(0);
            }

            cl::Buffer buf_sols_new(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * t.size, t.elements);
            kernel_resize.setArg(1, buf_sols_new);

            free(t.elements);

            cl::Buffer buf_num_sol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &t.numSolutions);
            kernel_resize.setArg(3, buf_num_sol);

            kernel_resize.setArg(4, table.minId);

            cl::Buffer buf_exp(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &(node.exponent));
            kernel_resize.setArg(5, buf_exp);

            cl_long error1 = 0, error2 = 0;
            cl_double range = table.maxId - table.minId;
            cl_long s = std::ceil(range / (1l << 31));
            for (cl_long i = 0; i < s; i++) {
                cl_long id1 = (1 << 31) * i;
                cl_long range = std::min((cl_long) 1 << 31, (cl_long) table.maxId - table.minId - (1 << 31) * i);
                error1 = queue.enqueueNDRangeKernel(kernel_resize, cl::NDRange(static_cast<size_t>(id1)), cl::NDRange(static_cast<size_t>(range)));
                error2 = queue.finish();
                if (error1 != 0 || error2 != 0) {
                    std::cerr << "\nResize 1 - OpenCL error: " << (error1 != 0 ? error1 : error2);
                    exit(1);
                }
            }
            queue.enqueueReadBuffer(buf_num_sol, CL_TRUE, 0, sizeof(cl_long), &(t.numSolutions));
            queue.enqueueReadBuffer(buf_exp, CL_TRUE, 0, sizeof(cl_long), &(node.exponent));

            t.elements = (cl_long *) malloc(sizeof(cl_long) * (t.numSolutions + 1 + nextSize));
            if (t.elements == NULL || errno == ENOMEM) {
                std::cerr << "\nOut of Memory\n";
                exit(0);
            }

            queue.enqueueReadBuffer(buf_sols_new, CL_TRUE, 0, sizeof(cl_long) * (t.numSolutions + 1 + nextSize), t.elements);
        }
        t.size = (t.numSolutions + 1 + nextSize);
        t.minId = std::min(table.minId, t.minId);
        t.maxId = std::max(table.maxId, t.maxId);
        table = t;
    }

    void Solver::combineTree(treeType &t, treeType &table, cl_long numVars) {
        std::cout << "combine tree forget" << std::endl;
        if (table.size > 0) {
            cl::Kernel kernel_resize = cl::Kernel(program, "combineTree");

            kernel_resize.setArg(0, numVars);

            cl::Buffer buf_sols_new(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * (t.numSolutions + table.numSolutions + 2), t.elements);
            kernel_resize.setArg(1, buf_sols_new);

            cl::Buffer buf_sols_old(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * table.size, table.elements);
            kernel_resize.setArg(2, buf_sols_old);
            free(table.elements);

            cl::Buffer buf_num_sol(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &t.numSolutions);
            kernel_resize.setArg(3, buf_num_sol);

            kernel_resize.setArg(4, table.minId);

            cl_long error1 = 0, error2 = 0;
            cl_double range = table.maxId - table.minId;
            cl_long s = std::ceil(range / (1l << 31));
            for (long i = 0; i < s; i++) {
                cl_long id1 = (1 << 31) * i;
                cl_long range = std::min((cl_long) 1 << 31, (cl_long) table.maxId - table.minId - (1 << 31) * i);
                error1 = queue.enqueueNDRangeKernel(kernel_resize, cl::NDRange(static_cast<size_t>(id1)), cl::NDRange(static_cast<size_t>(range)));
                error2 = queue.finish();
                if (error1 != 0 || error2 != 0) {
                    std::cerr << "\nResize 2 - OpenCL error: " << (error1 != 0 ? error1 : error2);
                    exit(1);
                }
            }
            queue.enqueueReadBuffer(buf_num_sol, CL_TRUE, 0, sizeof(cl_long), &t.numSolutions);
            queue.enqueueReadBuffer(buf_sols_new, CL_TRUE, 0, sizeof(cl_long) * (t.numSolutions + 1), t.elements);
        }
        t.minId = std::min(table.minId, t.minId);
        t.maxId = std::max(table.maxId, t.maxId);
    }

    void Solver::solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula, nodeType nextNode) {
        std::cout << "solve join" << std::endl;
        isSat = 0;
        this->numJoin++;
        cl::Kernel kernel = cl::Kernel(program, "solveJoin");
        cl::Buffer bufSolVars;
        if (node.variables.size() > 0) {
            bufSolVars = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * node.variables.size(), &node.variables[0]);
            kernel.setArg(3, bufSolVars);
        } else {
            kernel.setArg(3, NULL);
        }
        cl::Buffer bufSolVars1;
        if (edge1.variables.size() > 0) {
            bufSolVars1 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * edge1.variables.size(), &edge1.variables[0]);
            kernel.setArg(4, bufSolVars1);
        } else {
            kernel.setArg(4, NULL);
        }
        cl::Buffer bufSolVars2;
        if (edge2.variables.size() > 0) {
            bufSolVars2 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * edge2.variables.size(), &edge2.variables[0]);
            kernel.setArg(5, bufSolVars2);
        } else {
            kernel.setArg(5, NULL);
        }
        kernel.setArg(6, node.variables.size());
        kernel.setArg(7, edge1.variables.size());
        kernel.setArg(8, edge2.variables.size());
        cl::Buffer bufWeights;
        if (formula.variableWeights != nullptr) {
            bufWeights = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_double) * formula.numWeights, formula.variableWeights);
            kernel.setArg(16, bufWeights);
        } else {
            kernel.setArg(16, NULL);
        }

        node.exponent = CL_LONG_MIN;
        cl::Buffer buf_exp(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_long), &(node.exponent));
        kernel.setArg(19, buf_exp);

        kernel.setArg(18, pow(2, edge1.exponent + edge2.exponent));

        node.exponent = CL_LONG_MIN;
        cl_long usedMemory = sizeof(cl_long) * node.variables.size() * 3 + sizeof(cl_long) * edge1.variables.size() + sizeof(cl_long) * edge2.variables.size() + sizeof(cl_double) * formula.numWeights + sizeof(cl_double) * formula.numWeights;

        cl_long s = sizeof(cl_long);
        cl_long bagSizeNode = 1;

        if (maxBag > 0) {
            bagSizeNode = 1l << (cl_long) std::min(node.variables.size(), (size_t) maxBag);
        } else {
            if (solutionType == TREE) {
                if (nextNode == JOIN) {
                    bagSizeNode = std::min((cl_long) (maxMemoryBuffer / s / 2 - node.variables.size() * sizeof(cl_long) * 3), std::min((cl_long) std::min((memorySize - usedMemory - edge1.maxSize * s - edge2.maxSize * s) / s / 2, (memorySize - usedMemory) / 2 / 3 / s), 1l << node.variables.size()));
                } else if (nextNode == INTRODUCEFORGET) {
                    bagSizeNode = std::min((cl_long) (maxMemoryBuffer / s / 2 - node.variables.size() * sizeof(cl_long) * 3), std::min((cl_long) std::min((memorySize - usedMemory - edge1.maxSize * s - edge2.maxSize * s) / s / 2, (memorySize - usedMemory) / 2 / 2 / s), 1l << node.variables.size()));
                }
            } else if (solutionType == ARRAY) {
                bagSizeNode = 1l << (cl_long) std::min(node.variables.size(), (size_t) std::min(log2(maxMemoryBuffer / sizeof(cl_long)), log2(memorySize / sizeof(cl_long) / 3)));
            }
        }

        cl_long maxSize = std::ceil((1l << (node.variables.size())) * 1.0 / bagSizeNode);
        node.solution = new treeType[maxSize];
        if (node.solution == NULL || errno == ENOMEM) {
            std::cerr << "\nOut of Memory\n";
            exit(0);
        }
        node.bags = maxSize;
        for (cl_long a = 0, run = 0; a < node.bags; a++, run++) {
            node.solution[a].numSolutions = 0;
            node.solution[a].minId = run * bagSizeNode;
            node.solution[a].maxId = std::min(run * bagSizeNode + bagSizeNode, 1l << (node.variables.size()));
            node.solution[a].size = (node.solution[a].maxId - node.solution[a].minId) + node.variables.size();
            node.solution[a].elements = static_cast<cl_long *>(calloc(sizeof(cl_long), node.solution[a].size));
            if (node.solution[a].elements == NULL || errno == ENOMEM) {
                std::cerr << "\nOut of Memory\n";
                exit(0);
            }

            for (cl_long i = 0; i < node.solution[a].size; i++) {
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
                    bufSol1 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * edge1.solution[b].size, edge1.solution[b].elements);
                    kernel.setArg(1, bufSol1);
                } else {
                    kernel.setArg(1, NULL);
                }

                kernel.setArg(9, (b < edge1.bags) ? edge1.solution[b].minId : -1);
                kernel.setArg(10, (b < edge1.bags) ? edge1.solution[b].maxId : -1);
                kernel.setArg(14, (b < edge1.bags) ? edge1.solution[b].minId : 0);

                cl::Buffer bufSol2;
                if (b < edge2.bags && edge2.solution[b].elements != NULL) {
                    bufSol2 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_long) * edge2.solution[b].size, edge2.solution[b].elements);
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
		    std::cerr << error1 << std::endl;	
		    std::cerr << error2 << std::endl;	
                    std::cerr << "\nJoin - OpenCL error: " << (error1 != 0 ? error1 : error2) << "\n";
                    exit(1);
                }
            }
            queue.enqueueReadBuffer(bufsolBag, CL_TRUE, 0, sizeof(cl_long), &node.solution[a].numSolutions);
            if (node.solution[a].numSolutions == 0) {
                free(node.solution[a].elements);
                node.solution[a].elements = NULL;

                if (a > 0 and solutionType != ARRAY) {
                    node.solution[a - 1].maxId = node.solution[a].maxId;

                    node.bags--;
                    a--;
                }
            } else {
                queue.enqueueReadBuffer(bufSol, CL_TRUE, 0, sizeof(cl_long) * node.solution[a].size, node.solution[a].elements);
                this->isSat = 1;
                if (solutionType == TREE) {

                    if (a > 0 && node.solution[a - 1].elements != NULL && (node.solution[a].numSolutions + node.solution[a - 1].numSolutions + 2) < node.solution[a].size) {
                        cleanTree(node.solution[a], (bagSizeNode) * 2, node.variables.size(), node, node.solution[a - 1].numSolutions + 1);
                        combineTree(node.solution[a], node.solution[a - 1], node.variables.size());
                        node.solution[a - 1] = node.solution[a];

                        node.bags--;
                        a--;
                    } else if (a > 0 && node.solution[a - 1].elements == NULL) {
                        cleanTree(node.solution[a], (bagSizeNode) * 2, node.variables.size(), node, 0);
                        node.solution[a].minId = node.solution[a - 1].minId;
                        node.solution[a - 1] = node.solution[a];

                        node.bags--;
                        a--;
                    } else {
                        cleanTree(node.solution[a], (bagSizeNode) * 2, node.variables.size(), node, 0);
                    }
                    node.solution[a].size = node.solution[a].numSolutions + 1;
                    node.maxSize = std::max(node.maxSize, node.solution[a].size);
                } else if (solutionType == ARRAY) {
                    node.solution[a].size = bagSizeNode;
                    node.maxSize = std::max(node.maxSize, node.solution[a].size);
                }
            }
        }
        if (solutionType == ARRAY) {
            queue.enqueueReadBuffer(buf_exp, CL_TRUE, 0, sizeof(cl_long), &(node.exponent));
        }
        node.correction = edge1.correction + edge2.correction + edge1.exponent + edge2.exponent;
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
    }

    void Solver::solveIntroduceForget(satformulaType &formula, bagType &pnode, bagType &node, bagType &cnode, bool leaf, nodeType nextNode) {
    
        std::cout << "introduce forget" << std::endl;
        isSat = 0;
        std::vector<cl_long> fVars;
        std::set_intersection(node.variables.begin(), node.variables.end(), pnode.variables.begin(), pnode.variables.end(), std::back_inserter(fVars));
        std::vector<cl_long> iVars = node.variables;
        std::vector<cl_long> eVars = cnode.variables;

        node.variables = fVars;

        this->numIntroduceForget++;

        // get clauses which only contain iVars
        std::vector<cl_long> numVarsClause;
        std::vector<cl_long> clauses;
        cl_long numClauses = 0;
        for (cl_long i = 0; i < formula.clauses.size(); i++) {
            std::vector<cl_long> v;
            std::set_intersection(iVars.begin(), iVars.end(), formula.clauses[i].begin(), formula.clauses[i].end(), back_inserter(v), compVars);
            if (v.size() == formula.clauses[i].size()) {
                numClauses++;
                numVarsClause.push_back(formula.clauses[i].size());
                for (cl_long a = 0; a < formula.clauses[i].size(); a++) {
                    clauses.push_back(formula.clauses[i][a]);
                }
            }
        }
        
        node.exponent = CL_LONG_MIN;

        CudaBuffer<cl_long> buf_varsE = CudaBuffer<cl_long>(eVars);
        CudaBuffer<cl_long> buf_varsI = CudaBuffer<cl_long>(iVars);
        CudaBuffer<cl_long> buf_clauses = CudaBuffer<cl_long>(clauses);
        CudaBuffer<cl_long> buf_numVarsC = CudaBuffer<cl_long>(&numVarsClause.front(), numClauses);
        CudaBuffer<cl_double> buf_weights = CudaBuffer<cl_double>(formula.variableWeights, formula.numWeights);
        CudaBuffer<cl_long> buf_exponent = CudaBuffer<cl_long>(&(node.exponent), 1);

        size_t usedMemory = sizeof(cl_long) * eVars.size() + sizeof(cl_long) * iVars.size() * 3 + sizeof(cl_long) * (clauses.size()) + sizeof(cl_long) * (numClauses) + sizeof(cl_double) * formula.numWeights + sizeof(cl_long) * fVars.size() + sizeof(cl_double) * formula.numWeights;
        cl_long bagSizeForget = 1;
        cl_long s = sizeof(cl_long);


        if (maxBag > 0) {
            bagSizeForget = 1l << (cl_long) std::min(node.variables.size(), (size_t) maxBag);
        } else {
            if (solutionType == TREE) {
                if (nextNode == JOIN) {
                    bagSizeForget = std::min((cl_long) (maxMemoryBuffer / s / 2 - 3 * node.variables.size() * sizeof(cl_long)), std::min((cl_long) std::min((memorySize - usedMemory - cnode.maxSize * s) / s / 2, (memorySize - usedMemory) / 2 / 3 / s), 1l << node.variables.size()));
                } else if (nextNode == INTRODUCEFORGET) {
                    bagSizeForget = std::min((cl_long) (maxMemoryBuffer / s / 2 - 3 * node.variables.size() * sizeof(cl_long)), std::min((cl_long) std::min((memorySize - usedMemory - cnode.maxSize * s) / s / 2, (memorySize - usedMemory) / 2 / 2 / s), 1l << node.variables.size()));
                }
            } else if (solutionType == ARRAY) {
                bagSizeForget = 1l << (cl_long) std::min(node.variables.size(), (size_t) std::min(log2(maxMemoryBuffer / sizeof(cl_long)), log2(memorySize / sizeof(cl_long) / 3)));
            }
        }

        cl_long maxSize = std::ceil((1l << (node.variables.size())) * 1.0 / bagSizeForget);
        node.solution = new treeType[maxSize];
        if (node.solution == NULL || errno == ENOMEM) {
            std::cerr << "\nOut of Memory\n";
            exit(0);
        }
        node.bags = maxSize;
        for (cl_long a = 0, run = 0; a < node.bags; a++, run++) {
            node.solution[a].numSolutions = 0;
            node.solution[a].minId = run * bagSizeForget;
            node.solution[a].maxId = std::min(run * bagSizeForget + bagSizeForget, 1l << (node.variables.size()));
            node.solution[a].size = (node.solution[a].maxId - node.solution[a].minId) * 2 + node.variables.size();
            node.solution[a].elements = static_cast<cl_long *>(calloc(sizeof(cl_long), node.solution[a].size));
            if (node.solution[a].elements == NULL || errno == ENOMEM) {
                std::cerr << "\nOut of Memory\n";
                exit(0);
            }

            
            CudaBuffer<cl_long> buf_solsF = CudaBuffer<cl_long>(node.solution[a].elements, node.solution[a].size);
            CudaBuffer<cl_long> buf_varsF = CudaBuffer<cl_long>(fVars);
            CudaBuffer<cl_long> buf_solBag = CudaBuffer<cl_long>(&(node.solution[a].numSolutions), 1);
 
            for (cl_long b = 0; b < cnode.bags; b++) {
                if (cnode.solution[b].elements == NULL) {
                    continue;
                }


                CudaBuffer<cl_long> buf_solsE = CudaBuffer<cl_long>(cnode.solution[b].elements, cnode.solution[b].size);

                size_t threads = static_cast<size_t>(node.solution[a].maxId - node.solution[a].minId); 
                long id_offset = node.solution[a].minId;

                long combinations = (cl_long) pow(2, iVars.size() - fVars.size());
                
                // FIXME: offset onto global id 
                introduceForgetWrapper(
                    buf_solsF.device_mem,
                    buf_varsF.device_mem,
                    buf_solsE.device_mem,
                    buf_varsE.size(),
                    buf_varsE.device_mem,
                    combinations,
                    fVars.size(),
                    cnode.solution[b].minId,
                    cnode.solution[b].maxId,
                    node.solution[a].minId,
                    cnode.solution[b].minId,
                    buf_solBag.device_mem,
                    buf_varsI.size(),
                    buf_varsI.device_mem,
                    buf_clauses.device_mem,
                    buf_numVarsC.device_mem,
                    numClauses,
                    buf_weights.device_mem,
                    buf_exponent.device_mem, 
                    pow(2, cnode.exponent),
                    threads,
                    id_offset
                );
            } 

            buf_solBag.read(&node.solution[a].numSolutions);

            std::cout << "num solutions: " << node.solution[a].numSolutions << std::endl;
            if (node.solution[a].numSolutions == 0) {
                free(node.solution[a].elements);
                node.solution[a].elements = NULL;

                if (a > 0 and solutionType != ARRAY) {
                    node.solution[a - 1].maxId = node.solution[a].maxId;

                    node.bags--;
                    a--;
                }
            } else {
                this->isSat = 1;

                if (solutionType == TREE) {
                    if (node.variables.size() == 0) {
                        node.solution[a].numSolutions--;
                    }
                    free(node.solution[a].elements);
                    if (a > 0 && node.solution[a - 1].elements != nullptr && (node.solution[a].numSolutions + node.solution[a - 1].numSolutions + 2) < node.solution[a].size) {
                        node.solution[a].elements = (cl_long *) malloc(sizeof(cl_long) * (node.solution[a].numSolutions + node.solution[a - 1].numSolutions + 2));
                        if (node.solution[a].elements == NULL || errno == ENOMEM) {
                            std::cerr << "\nOut of Memory\n";
                            exit(0);
                        }

                        long solFCount = sizeof(cl_long) * (node.solution[a].numSolutions + node.solution[a - 1].numSolutions + 2);
                        cudaMemcpy(node.solution[a].elements, buf_solsF.device_mem, solFCount, cudaMemcpyDeviceToHost);

                        combineTree(node.solution[a], node.solution[a - 1], node.variables.size());
                        node.solution[a - 1] = node.solution[a];

                        node.bags--;
                        a--;
                    } else {
                        node.solution[a].elements = (cl_long *) malloc(sizeof(cl_long) * (node.solution[a].numSolutions + 1));
                        if (node.solution[a].elements == NULL || errno == ENOMEM) {
                            std::cerr << "\nOut of Memory\n";
                            exit(0);
                        }
                        long solFCount = sizeof(cl_long) * (node.solution[a].numSolutions + 1);
                        cudaMemcpy(node.solution[a].elements, buf_solsF.device_mem, solFCount, cudaMemcpyDeviceToHost);

                        if (a > 0 && node.solution[a - 1].elements == NULL) {
                            node.solution[a].minId = node.solution[a - 1].minId;
                            node.solution[a - 1] = node.solution[a];

                            node.bags--;
                            a--;
                        }
                    }
                    node.solution[a].size = node.solution[a].numSolutions + 1;
                    node.maxSize = std::max(node.maxSize, node.solution[a].size);
                } else if (solutionType == ARRAY) {
                    node.solution[a].size = bagSizeForget;

                    cudaMemcpy(node.solution[a].elements, buf_solsF.device_mem, sizeof(long) * bagSizeForget, cudaMemcpyDeviceToHost);
                }
            }
        }

        buf_exponent.read(&(node.exponent));
        node.correction = cnode.correction + cnode.exponent;
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
    }
}
