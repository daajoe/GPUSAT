#include <algorithm>
#include <cmath>
#include <iostream>
#include <solver.h>
#include <errno.h>
#include <cuda.h>
#include <memory>
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
   code = cudaGetLastError();
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert last error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

extern void introduceForgetWrapper(long *solsF,  long *varsF,  long *solsE, long numVE,  long *varsE, long combinations, long numVF, long minIdE, long maxIdE, long startIDF, long startIDE,  long *sols, long numVI,  long *varsI,  long *clauses,  long *numVarsC, long numclauses,  double *weights,  long *exponent, double value, size_t threads, long id_offset);

extern void solveJoinWrapper( long *solutions,  long *edge1,  long *edge2,  long *variables,  long *edgeVariables1,  long *edgeVariables2, long numV, long numVE1, long numVE2, long minId1, long maxId1, long minId2, long maxId2, long startIDNode, long startIDEdge1, long startIDEdge2,  double *weights,  long *sols, double value,  long *exponent, size_t threads, long id_offset);

extern void resizeWrapper(long numVars, long *tree, double *solutions_old, long *treeSize, long startId, long *exponent, size_t threads, long id_offset);

extern void combineTreeWrapper(long numVars, long *tree, double *solutions_old, long *treeSize, long startId, size_t threads, long id_offset);

namespace gpusat {

    double getCount(long id, long *tree, long numVars) {
        ulong nextId = 0;
        for (ulong i = 0; i < numVars; i++) {
            // upper
            if ((id >> (numVars - i - 1)) & 1) {
                nextId = ((ulong)tree[nextId] & 0xFFFFFFFF00000000) >> 32;
            } else {
                nextId = ((ulong)tree[nextId] & 0x00000000FFFFFFFF);
            }
            if (nextId == 0) {
                return 0.0;
            }
        }
        return *reinterpret_cast<double*>(&tree[nextId]);
    }

    size_t hashSubtree(long* elements, long element, int variables) {
        if (element == 0) return 0;
        // index cell
        if (variables > 0) {
            uint32_t lower = element & 0x00000000FFFFFFFF;
            uint32_t upper = (element & 0xFFFFFFFF00000000) >> 32;
            size_t hash = 0;
            if (lower) {
                hash ^= 1;
                hash ^= (hashSubtree(elements, elements[lower], variables - 1) << 1);
            }
            if (upper) {
                hash ^= 2;
                hash ^= (hashSubtree(elements, elements[upper], variables - 1) << 1);
            }
            return hash;
        } else {
            return (size_t)element;
        }
    }
    size_t hashSolution(treeType *t, int vars) {
        if (t->elements == NULL) {
            return 0;
        }
        return hashSubtree(t->elements, t->elements[0], vars);
    }

    size_t hashArray(long *elements, long size) {
        size_t h = 0;
        for (int i=0; i < size; i++) {
            h ^= (elements[i] << 1);
        }
        return h;
    }

    void enumerateSubtree(char* prefix, long* elements, long element, int variables) {
        if (element == 0) return;
        // index cell
        if (variables > 0) {
            uint32_t lower = element & 0x00000000FFFFFFFF;
            uint32_t upper = (element & 0xFFFFFFFF00000000) >> 32;
            char new_prefix[256] = {0};
            if (lower) {
                sprintf(new_prefix, "%s %d", prefix, 0);
                enumerateSubtree(new_prefix, elements, elements[lower], variables - 1);
            }
            if (upper) {
                sprintf(new_prefix, "%s %d", prefix, 1);
                enumerateSubtree(new_prefix, elements, elements[upper], variables - 1);
            }
        // value cell
        } else {
            std::cerr << prefix << " : " << reinterpret_cast<double &>(element) << std::endl;
        }
    }
    void enumerateSolutions(treeType *t) {
        int vars = std::ceil(log2(t->maxId - t->minId));
        std::cerr << "min ID: " << t->minId << " max ID: " << t->maxId << " -> vars: " << vars << std::endl;
        if (t->elements == NULL) {
            std::cerr << "empty tree!" << std::endl;
            return;
        }
        enumerateSubtree("", t->elements, t->elements[0], vars);
    }

    size_t treeTypeHash(treeType *input, int vars) {
        size_t h = 0;
        if (input == NULL) {
            return h;
        }
        h = h ^ (hashSolution(input, vars) << 1);
        h = h ^ (input->minId << 1);
        h = h ^ (input->maxId << 1);
        h = h ^ (input->numSolutions << 1);
        return h;
    }
    size_t bagTypeHash(bagType *input) {
        size_t h = 0;
        if (input == NULL) {
            return h;
        }
        h = h ^ (input->correction << 1);
        h = h ^ (input->exponent << 1);
        h = h ^ (input->id << 1);
        for (cl_long var : input->variables) {
            h = h ^ (var << 1);
        }
        for (bagType *edge : input->edges) {
            h = h ^ (bagTypeHash(edge) << 1);
        }  
        h = h ^ (input->bags << 1);
        for (int i=0; i < input->bags; i++) {
            h = h ^ (treeTypeHash(&input->solution[i], input->variables.size()) << 1);
        }
        h = h ^ (input->maxSize << 1);
        return h;
    }
    template <typename T>
    class CudaBuffer {
        private:
            size_t buf_size;

            // prevent c++ from copy assignment.
            // learned this the hard way...
            CudaBuffer(const CudaBuffer& other);
            CudaBuffer& operator=(const CudaBuffer& other);

        public:
            T* device_mem;
            // creates a buffer with size 0.
            CudaBuffer();
            /// Create an on-device array of T with given length
            CudaBuffer(T* from, size_t length);
            /// Copy a vector to the device.
            /// If the vector is empty, the memory pointer is NULL. 
            CudaBuffer(std::vector<T> &vec);
            /// Copy on-device array to `to`
            void read(T* to);
            /// Length of the buffer
            size_t size();

            ~CudaBuffer();
    };

    template <typename T>
    CudaBuffer<T>::CudaBuffer(T* from, size_t length) {
        T* mem = NULL;
        if (from == NULL) {
            this->buf_size = 0;
        } else {
            gpuErrchk(cudaMalloc((void**)&mem, sizeof(T) * length));
            gpuErrchk(cudaMemcpy(mem, from, sizeof(T) * length, cudaMemcpyHostToDevice));
            this->buf_size = length;
        }
        this->device_mem = mem;
    }

    template <typename T>
    CudaBuffer<T>::CudaBuffer(std::vector<T> &vec) {
        T* mem = NULL;
        if (vec.size()) {
            gpuErrchk(cudaMalloc((void**)&mem, sizeof(T) * vec.size()));
            gpuErrchk(cudaMemcpy(mem, &vec[0], sizeof(T) * vec.size(), cudaMemcpyHostToDevice));
            this->buf_size = vec.size();
        } else {
            this->buf_size = 0;
        }
        this->device_mem = mem;
    }

    template <typename T>
    CudaBuffer<T>::CudaBuffer() {
        this->device_mem = NULL;
        this->buf_size = 0;
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
        if (this->device_mem) {
            gpuErrchk(cudaFree(this->device_mem));
        }
        this->device_mem = NULL;
    }


    void Solver::solveProblem(treedecType &decomp, satformulaType &formula, bagType &node, bagType &pnode, nodeType lastNode) {

        std::cerr << "solve problem. isSAT: " << isSat << " edges:" << node.edges.size() << std::endl;
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
        std::cerr << "clean tree input hash: " << hashArray(table.elements, table.size) << " table size: " << table.size << " numVars: " << numVars << std::endl;
        treeType t;
        t.numSolutions = 0;
        t.size = size + numVars;
        t.minId = table.minId;
        t.maxId = table.maxId;
        if (table.size > 0) {

            CudaBuffer<cl_double> buf_sols_old((cl_double*)table.elements, table.size);
            free(table.elements);

            t.elements = static_cast<cl_long *>(calloc(sizeof(cl_long), size + numVars * 3));
            if (t.elements == NULL || errno == ENOMEM) {
                std::cerr << "\nOut of Memory\n";
                exit(0);
            }
            size_t mfree, total;

            CudaBuffer<cl_long> buf_sols_new(t.elements, t.size);
            std::cerr << "clean tree new sols size: " << t.size << std::endl;
            free(t.elements);

            CudaBuffer<cl_long> buf_num_sol(&(t.numSolutions), 1);

            CudaBuffer<cl_long> buf_exp(&(node.exponent), 1);

            cl_long error1 = 0, error2 = 0;
            cl_double range = table.maxId - table.minId;
            cl_long s = std::ceil(range / (1l << 31));
            for (cl_long i = 0; i < s; i++) {
                cl_long id1 = (1 << 31) * i;
                cl_long range = std::min((cl_long) 1 << 31, (cl_long) table.maxId - table.minId - (1 << 31) * i);
                resizeWrapper(
                    numVars,
                    buf_sols_new.device_mem,
                    buf_sols_old.device_mem,
                    buf_num_sol.device_mem,
                    table.minId,
                    buf_exp.device_mem,
                    range,
                    id1
                ); 
            }
            // actually the tree size
            buf_num_sol.read(&(t.numSolutions));
            buf_exp.read(&(node.exponent));
            std::cerr << "clean tree num solutions: " << t.numSolutions << std::endl;

            t.elements = (cl_long *) malloc(sizeof(cl_long) * (t.numSolutions + 1 + nextSize));
            if (t.elements == NULL || errno == ENOMEM) {
                std::cerr << "\nOut of Memory\n";
                exit(0);
            }

            std::cerr << "clean tree new sols copy size: " << (t.numSolutions + 1 + nextSize) << std::endl;
            // why not qual?
            gpuErrchk(cudaMemcpy(t.elements, buf_sols_new.device_mem, sizeof(cl_long) * (t.numSolutions + 1 + nextSize), cudaMemcpyDeviceToHost));
        }
        t.size = (t.numSolutions + 1 + nextSize);
        t.minId = std::min(table.minId, t.minId);
        t.maxId = std::max(table.maxId, t.maxId);
        table = t;
        std::cerr << "clean tree output hash: " << hashSolution(&t, numVars) << std::endl;
    }

    void Solver::combineTree(treeType &t, treeType &table, cl_long numVars) {
        std::cerr << "combine tree" << treeTypeHash(&t, numVars) << " " << treeTypeHash(&table, numVars) << std::endl;
        if (table.size > 0) {

            CudaBuffer<cl_long> buf_sols_new(t.elements, t.numSolutions + table.numSolutions + 2);
            CudaBuffer<cl_double> buf_sols_old((double*)table.elements, t.size);

            free(table.elements);

            CudaBuffer<cl_long> buf_num_sol(&t.numSolutions, 1);

            cl_long error1 = 0, error2 = 0;
            cl_double range = table.maxId - table.minId;
            cl_long s = std::ceil(range / (1l << 31));
            for (long i = 0; i < s; i++) {
                cl_long id1 = (1 << 31) * i;
                cl_long range = std::min((cl_long) 1 << 31, (cl_long) table.maxId - table.minId - (1 << 31) * i);
                combineTreeWrapper(
                    numVars,
                    buf_sols_new.device_mem,
                    buf_sols_old.device_mem,
                    buf_num_sol.device_mem,
                    table.minId,
                    range,
                    id1
                );
            }
            buf_num_sol.read(&t.numSolutions);
            gpuErrchk(cudaMemcpy(t.elements, buf_sols_new.device_mem, sizeof(cl_long) * (t.numSolutions + 1), cudaMemcpyDeviceToHost));
        }
        t.minId = std::min(table.minId, t.minId);
        t.maxId = std::max(table.maxId, t.maxId);
        std::cerr << "combine tree output hash: " << hashSubtree(t.elements, t.elements[0], numVars) << std::endl;
    }

    void Solver::solveJoin(bagType &node, bagType &edge1, bagType &edge2, satformulaType &formula, nodeType nextNode) {
        isSat = 0;
        this->numJoin++;
         
        CudaBuffer<cl_long> buf_solVars(node.variables);
        CudaBuffer<cl_long> buf_solVars1(edge1.variables);
        CudaBuffer<cl_long> buf_solVars2(edge2.variables);

        
        std::unique_ptr<CudaBuffer<cl_double>> buf_weights( std::make_unique<CudaBuffer<cl_double>>() );
        if (formula.variableWeights != nullptr) {
            buf_weights = std::make_unique<CudaBuffer<cl_double>>(formula.variableWeights, formula.numWeights);
        }
        
        node.exponent = CL_LONG_MIN;
        CudaBuffer<cl_long> buf_exponent(&(node.exponent), 1);

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

            CudaBuffer<cl_long> buf_sol(node.solution[a].elements, node.solution[a].size);
            CudaBuffer<cl_long> buf_solBag(&(node.solution[a].numSolutions), 1);

            for (cl_long b = 0; b < std::max(edge1.bags, edge2.bags); b++) {
                
                std::unique_ptr<CudaBuffer<cl_long>> buf_sol1( std::make_unique<CudaBuffer<cl_long>>() );
                if (b < edge1.bags && edge1.solution[b].elements != NULL) {
                    buf_sol1 = std::make_unique<CudaBuffer<cl_long>>(edge1.solution[b].elements, edge1.solution[b].size);
                }

                std::unique_ptr<CudaBuffer<cl_long>> buf_sol2( std::make_unique<CudaBuffer<cl_long>>() );
                if (b < edge2.bags && edge2.solution[b].elements != NULL) {
                    buf_sol2 = std::make_unique<CudaBuffer<cl_long>>(edge2.solution[b].elements, edge2.solution[b].size);
                }

                long id_offset = node.solution[a].minId;
                size_t threads = static_cast<size_t>(node.solution[a].maxId - node.solution[a].minId);
                std::cerr << "thread offset: " << id_offset << " threads " << threads << std::endl;
                
                solveJoinWrapper(
                    buf_sol.device_mem,
                    buf_sol1->device_mem,
                    buf_sol2->device_mem,
                    buf_solVars.device_mem,
                    buf_solVars1.device_mem,
                    buf_solVars2.device_mem,
                    node.variables.size(),
                    edge1.variables.size(),
                    edge2.variables.size(),
                    (b < edge1.bags) ? edge1.solution[b].minId : -1,
                    (b < edge1.bags) ? edge1.solution[b].maxId : -1,
                    (b < edge2.bags) ? edge2.solution[b].minId : -1,
                    (b < edge2.bags) ? edge2.solution[b].maxId : -1,
                    node.solution[a].minId,
                    (b < edge1.bags) ? edge1.solution[b].minId : 0,
                    (b < edge2.bags) ? edge2.solution[b].minId : 0,
                    buf_weights->device_mem,
                    buf_solBag.device_mem,
                    pow(2, edge1.exponent + edge2.exponent),
                    buf_exponent.device_mem,
                    threads,
                    id_offset
                );
            }

            buf_solBag.read(&(node.solution[a].numSolutions));
            std::cerr << "num solutions (join): " << node.solution[a].numSolutions << std::endl;
            if (node.solution[a].numSolutions == 0) {
                free(node.solution[a].elements);
                node.solution[a].elements = NULL;

                if (a > 0 and solutionType != ARRAY) {
                    node.solution[a - 1].maxId = node.solution[a].maxId;

                    node.bags--;
                    a--;
                }
            } else {
                // node.elements is an array here
                buf_sol.read(node.solution[a].elements);
                this->isSat = 1;
                if (solutionType == TREE) {

                    if (a > 0 && node.solution[a - 1].elements != NULL && (node.solution[a].numSolutions + node.solution[a - 1].numSolutions + 2) < node.solution[a].size) {
                        std::cerr << "first branch" << std::endl;
                        cleanTree(node.solution[a], (bagSizeNode) * 2, node.variables.size(), node, node.solution[a - 1].numSolutions + 1);
                        combineTree(node.solution[a], node.solution[a - 1], node.variables.size());
                        node.solution[a - 1] = node.solution[a];

                        node.bags--;
                        a--;
                    } else if (a > 0 && node.solution[a - 1].elements == NULL) {
                        std::cerr << "second branch" << std::endl;
                        cleanTree(node.solution[a], (bagSizeNode) * 2, node.variables.size(), node, 0);
                        node.solution[a].minId = node.solution[a - 1].minId;
                        node.solution[a - 1] = node.solution[a];

                        node.bags--;
                        a--;
                    } else {
                        std::cerr << "simple clean tree" << std::endl;
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
            buf_exponent.read(&(node.exponent));
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

        CudaBuffer<cl_long> buf_varsE(eVars);
        CudaBuffer<cl_long> buf_varsI(iVars);
        CudaBuffer<cl_long> buf_clauses(clauses);
        CudaBuffer<cl_long> buf_numVarsC(&numVarsClause[0], numClauses);
        CudaBuffer<cl_double> buf_weights(formula.variableWeights, formula.numWeights);
        CudaBuffer<cl_long> buf_exponent(&(node.exponent), 1);

        size_t usedMemory = sizeof(cl_long) * eVars.size() + sizeof(cl_long) * iVars.size() * 3 + sizeof(cl_long) * (clauses.size()) + sizeof(cl_long) * (numClauses) + sizeof(cl_double) * formula.numWeights + sizeof(cl_long) * fVars.size() + sizeof(cl_double) * formula.numWeights;
        cl_long bagSizeForget = 1;
        cl_long s = sizeof(cl_long);

        std::cerr << "iVars: " << iVars.size() << std::endl;
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

            
            CudaBuffer<cl_long> buf_solsF(node.solution[a].elements, node.solution[a].size);
            CudaBuffer<cl_long> buf_varsF(fVars);
            CudaBuffer<cl_long> buf_solBag(&(node.solution[a].numSolutions), 1);
 
            for (cl_long b = 0; b < cnode.bags; b++) {
                if (cnode.solution[b].elements == NULL) {
                    continue;
                }


                std::unique_ptr<CudaBuffer<cl_long>> buf_solsE( std::make_unique<CudaBuffer<cl_long>>() );
                if (!leaf) {
                    buf_solsE = std::make_unique<CudaBuffer<cl_long>>(cnode.solution[b].elements, cnode.solution[b].size);
                }

                size_t threads = static_cast<size_t>(node.solution[a].maxId - node.solution[a].minId); 
                long id_offset = node.solution[a].minId;

                long combinations = (cl_long) pow(2, iVars.size() - fVars.size());
                
                size_t free, total;
                gpuErrchk(cudaMemGetInfo(&free, &total));
                std::cerr << "mem usage:" << free << " free, " << total <<" total." << std::endl;
                // FIXME: offset onto global id 
                introduceForgetWrapper(
                    buf_solsF.device_mem,
                    buf_varsF.device_mem,
                    buf_solsE->device_mem,
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
            buf_solBag.read(&(node.solution[a].numSolutions));

            std::cerr << "num solutions: " << node.solution[a].numSolutions << std::endl;
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
        std::cerr << "exponent: " << node.exponent << std::endl;
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
