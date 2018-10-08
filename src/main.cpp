#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <sstream>
#include <fstream>
#include <regex>
#include <math.h>
#include <chrono>
#include <types.h>
#include <gpusatparser.h>
#include <gpusautils.h>
#include <sys/stat.h>
#include <numeric>
#include <solver.h>
#include <CLI11.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <gpusatpreprocessor.h>

using namespace gpusat;

long long int getTime();

bool isPrimalGraph(satformulaType *formula, treedecType *decomp);

int main(int argc, char *argv[]) {
    long long int time_total = getTime();
    std::stringbuf treeD, sat;
    std::string inputLine;
    std::string formulaDir;
    std::string decompDir;
    int combineWidth = 12;
    std::string kernelPath = "./kernel/";
    graphTypes graph = graphTypes::NONE;
    bool factR, cpu, weighted, nvidia, amd;
    CLI::App app{};

    std::string filename = "default";
    app.add_option("-s,--formula", formulaDir, "path to the file containing the sat formula")->required();
    app.add_option("-f,--decomposition", decompDir, "path to the file containing the tree decomposition")->set_default_str("");
    app.add_option("-w,--combineWidth", combineWidth, "maximum width to combine bags of the decomposition")->set_default_str("10");
    app.add_option("-c,--kernelDir", kernelPath, "directory containing the kernel files")->set_default_str("./kernel/");
    app.add_flag("--noFactRemoval", factR, "deactivate fact removal optimization");
    app.add_flag("--CPU", cpu, "run the solver on the cpu");
    app.add_flag("--NVIDIA", nvidia, "run the solver on an NVIDIA device");
    app.add_flag("--AMD", amd, "run the solver on an AMD device");
    app.add_flag("--weighted", weighted, "use weighted model count");


    CLI11_PARSE(app, argc, argv);

    if (decompDir.compare("") != 0) {
        std::ifstream fileIn(decompDir);
        while (getline(fileIn, inputLine)) {
            treeD.sputn(inputLine.c_str(), inputLine.size());
            treeD.sputn("\n", 1);
        }
    } else {
        while (getline(std::cin, inputLine)) {
            treeD.sputn(inputLine.c_str(), inputLine.size());
            treeD.sputn("\n", 1);
        }
    }

    std::ifstream fileIn(formulaDir);
    while (getline(fileIn, inputLine)) {
        sat.sputn(inputLine.c_str(), inputLine.size());
        sat.sputn("\n", 1);
    }

    std::cout << "{\n";

    long long int time_parsing = getTime();
    CNFParser cnfParser(weighted);
    TDParser tdParser(combineWidth, factR);
    std::string satString = sat.str();
    std::string treeDString = treeD.str();
    if (satString.size() < 10) {
        std::cerr << "Error: SAT formula\n";
        exit(EXIT_FAILURE);
    }
    if (treeDString.size() < 9) {
        std::cerr << "Error: tree decomposition\n";
        exit(EXIT_FAILURE);
    }
    //parse the sat formula
    satformulaType satFormula = cnfParser.parseSatFormula(sat.str());
    //parse the tree decomposition
    treedecType treeDecomp = tdParser.parseTreeDecomp(treeD.str(), satFormula, graph);

    std::cout << "    \"pre Width\": " << tdParser.preWidth;
    std::cout << "\n    ,\"pre Cut Set Size\": " << tdParser.preCut;
    std::cout << "\n    ,\"pre Join Size\": " << tdParser.preJoinSize;
    std::cout << "\n    ,\"pre Bags\": " << tdParser.preNumBags;
    std::cout.flush();

    if (satFormula.clauses.size() == satFormula.numVars && satFormula.numVars == treeDecomp.numVars) {
        if (isPrimalGraph(&satFormula, &treeDecomp)) {
            graph = graphTypes::PRIMAL;
        } else {
            graph = graphTypes::DUAL;
        }
    } else if (satFormula.clauses.size() == treeDecomp.numVars) {
        // decomposition is of the dual
        graph = graphTypes::DUAL;
    } else if (satFormula.numVars == treeDecomp.numVars) {
        // decomposition is of the incidence graph
        graph = graphTypes::PRIMAL;
    } else if (satFormula.numVars + satFormula.clauses.size() == treeDecomp.numVars) {
        // decomposition is of the primal graph
        graph = graphTypes::INCIDENCE;
    } else {
        std::cerr << "Error: Unknown graph type\n";
        exit(EXIT_FAILURE);
    }

    // remove facts form decomp and formula
    if (!factR) {
        Preprocessor::preprocessFacts(treeDecomp, satFormula, graph, tdParser.defaultWeight);
        if (satFormula.unsat) {
            time_parsing = getTime() - time_parsing;
            time_total = getTime() - time_total;
            std::cout << "\n    ,\"Model Count\": " << 0;
            std::cout << "\n    ,\"Time\":{";
            std::cout << "\n        \"Solving\": " << 0;
            std::cout << "\n        ,\"Parsing\": " << ((float) time_parsing) / 1000;
            std::cout << "\n        ,\"Build_Kernel\": " << 0;
            std::cout << "\n        ,\"Generate_Model\": " << 0;
            std::cout << "\n        ,\"Init_OpenCL\": " << 0;
            std::cout << "\n        ,\"Total\": " << ((float) time_total) / 1000;
            std::cout << "\n    }";
            std::cout << "\n    ,\"Statistics\":{";
            std::cout << "\n        \"Num Join\": " << 0;
            std::cout << "\n        ,\"Num Forget\": " << 0;
            std::cout << "\n        ,\"Num Introduce\": " << 0;
            std::cout << "\n        ,\"Num Leaf\": " << 0;
            std::cout << "\n    }";
            std::cout << "\n}\n";
            exit(20);
        }
    }
    // combine small bags
    Preprocessor::preprocessDecomp(&treeDecomp.bags[0], combineWidth);

    tdParser.iterateDecompPost(treeDecomp.bags[0]);
    tdParser.postNumBags = treeDecomp.bags.size();

    time_parsing = getTime() - time_parsing;

    std::cout << "\n    ,\"post Width\": " << tdParser.postWidth;
    std::cout << "\n    ,\"post Cut Set Size\": " << tdParser.postCut;
    std::cout << "\n    ,\"post Join Size\": " << tdParser.postJoinSize;
    std::cout << "\n    ,\"post Bags\": " << tdParser.postNumBags;
    std::cout.flush();

    std::vector<cl::Platform> platforms;
    cl::Context context;
    std::vector<cl::Device> devices;
    cl::CommandQueue queue;
    cl::Program program;
    cl_long memorySize = 0;

    try {
        long long int time_init_opencl = getTime();
        cl::Platform::get(&platforms);
        std::vector<cl::Platform>::iterator iter;
        for (iter = platforms.begin(); iter != platforms.end(); ++iter) {
            if (nvidia && amd) {
                if (strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "NVIDIA Corporation") && strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "Advanced Micro Devices, Inc.")) {
                    continue;
                }
            } else if (nvidia) {
                if (strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "NVIDIA Corporation")) {
                    continue;
                }
            } else if (amd) {
                if (strcmp((*iter).getInfo<CL_PLATFORM_VENDOR>().c_str(), "Advanced Micro Devices, Inc.")) {
                    continue;
                }
            }
            cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties) (*iter)(), 0};
            if (cpu) {
                context = cl::Context(CL_DEVICE_TYPE_CPU, cps);
            } else {
                context = cl::Context(CL_DEVICE_TYPE_GPU, cps);
            }
            cl_int err;
            devices = context.getInfo<CL_CONTEXT_DEVICES>(&err);
            if (err == CL_SUCCESS) {
                queue = cl::CommandQueue(context, devices[0]);
                memorySize = devices[0].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()/262144;
                memorySize = devices[0].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
                break;
            }
        }
        if (iter == platforms.end()) {
            std::cout << "\nERROR: no GPU found!";
            exit(1);
        }
        time_init_opencl = getTime() - time_init_opencl;

        long long int time_build_kernel = getTime();
        struct stat buffer;
        std::string binPath;

        switch (graph) {
            case graphTypes::DUAL:
                binPath = kernelPath + "SAT_d_d.clbin";
                break;
            case graphTypes::PRIMAL:
                binPath = kernelPath + "SAT_d_p.clbin";
                break;
            case graphTypes::INCIDENCE:
                binPath = kernelPath + "SAT_d_i.clbin";
                break;
            default:
                break;
        }

#ifndef DEBUG
        if (stat(binPath.c_str(), &buffer) != 0) {
#endif
        //create kernel binary if it doesn't exist
        std::string sourcePath;

        switch (graph) {
            case graphTypes::DUAL:
                sourcePath = kernelPath + "SAT_d_dual.cl";
                break;
            case graphTypes::PRIMAL:
                sourcePath = kernelPath + "SAT_d_primal.cl";
                break;
            case graphTypes::INCIDENCE:
                sourcePath = kernelPath + "SAT_d_inci.cl";
                break;
        }
        // read source file
        std::string kernelStr = GPUSATUtils::readFile(sourcePath);
        cl::Program::Sources sources(1, std::make_pair(kernelStr.c_str(), kernelStr.length()));
        program = cl::Program(context, sources);
        program.build(devices);

        const std::vector<size_t> binSizes = program.getInfo<CL_PROGRAM_BINARY_SIZES>();
        std::vector<char> binData((unsigned long long int) std::accumulate(binSizes.begin(), binSizes.end(), 0));
        char *binChunk = &binData[0];

        std::vector<char *> binaries;
        for (const size_t &binSize : binSizes) {
            binaries.push_back(binChunk);
            binChunk += binSize;
        }

        // write binaries
        program.getInfo(CL_PROGRAM_BINARIES, &binaries[0]);
        std::ofstream binaryfile(binPath.c_str(), std::ios::binary);
        for (unsigned int i = 0; i < binaries.size(); ++i)
            binaryfile.write(binaries[i], binSizes[i]);
        binaryfile.close();
#ifndef DEBUG
        } else {
            //load kernel binary

            long size = 0;
            cl_int err;
            std::string kernelStr = GPUSATUtils::readBinary(binPath);
            cl::Program::Binaries bins(1, std::make_pair((const void *) kernelStr.data(), kernelStr.size()));
            program = cl::Program(context, devices, bins, nullptr, &err);
            program.build(devices);
        }
#endif
        time_build_kernel = getTime() - time_build_kernel;

        Solver *sol;
        bagType next;
        switch (graph) {
            case graphTypes::PRIMAL:
                sol = new Solver_Primal(context, queue, program, memorySize);
                next.variables.assign(treeDecomp.bags[0].variables.begin(), treeDecomp.bags[0].variables.begin() + std::min((cl_long) treeDecomp.bags[0].variables.size(), (cl_long) 12));
                break;
        }
        long long int time_solving = getTime();
        (*sol).solveProblem(treeDecomp, satFormula, treeDecomp.bags[0], next);
        time_solving = getTime() - time_solving;

        std::cout << "\n    ,\"Num Join\": " << sol->numJoin;
        std::cout << "\n    ,\"Num Introduce Forget\": " << sol->numIntroduceForget;
        std::cout << "\n    ,\"max Table Size\": " << sol->maxTableSize;
        std::cout << "\n    ,\"is SAT\": " << sol->isSat;
        cl_long solLeafs = 0;
        for (cl_long a = 0; a < treeDecomp.bags[0].bags; a++) {
            solLeafs += treeDecomp.bags[0].solution[a].numSolutions;
        }
        std::cout << "\n    ,\"solution Leafs\": " << solLeafs;

        //sum up last node solutions
        long long int time_model = getTime();
        boost::multiprecision::cpp_bin_float_100 sols = 0.0;
        if ((*sol).isSat > 0) {
            for (cl_long a = 0; a < treeDecomp.bags[0].bags; a++) {
                for (cl_long i = treeDecomp.bags[0].solution[a].minId; i < treeDecomp.bags[0].solution[a].maxId; i++) {
                    if (treeDecomp.bags[0].solution[a].elements != nullptr) {
                        sols = sols + GPUSATUtils::getCount(i, treeDecomp.bags[0].solution[a].elements, treeDecomp.bags[0].variables.size());
                    }
                }
                if (treeDecomp.bags[0].solution[a].elements != NULL)
                    delete[] treeDecomp.bags[0].solution[a].elements;
            }
            if (!weighted && graph != graphTypes::DUAL) {
                boost::multiprecision::cpp_bin_float_100 base = 0.78, exponent = satFormula.numVars;
                sols = sols / pow(base, exponent);
            } else if (graph != graphTypes::DUAL) {
                sols = sols * tdParser.defaultWeight;
            }

            if (graph == graphTypes::DUAL) {
                std::set<cl_long> varSet;
                for (int j = 0; j < satFormula.clauses.size(); ++j) {
                    for (int i = 0; i < satFormula.clauses[j].size(); ++i) {
                        varSet.insert(satFormula.clauses[j][i]);
                    }
                }
                for (int i = 1; i <= satFormula.numVars; ++i) {
                    if (varSet.find(i) == varSet.end() && varSet.find(-i) == varSet.end())
                        sols *= 2;
                }
            }
            char buf[128];

            std::cout << std::setprecision(20) << "\n    ,\"Model Count\": " << sols;

        } else {
            std::cout << "\n    ,\"Model Count\": " << 0;
        }
        time_model = getTime() - time_model;
        time_total = getTime() - time_total;
        std::cout.precision(6);
        std::cout << "\n    ,\"Time\":{";
        std::cout << "\n        \"Solving\": " << ((float) time_solving) / 1000;
        std::cout << "\n        ,\"Parsing\": " << ((float) time_parsing) / 1000;
        std::cout << "\n        ,\"Build_Kernel\": " << ((float) time_build_kernel) / 1000;
        std::cout << "\n        ,\"Generate_Model\": " << ((float) time_model) / 1000;
        std::cout << "\n        ,\"Init_OpenCL\": " << ((float) time_init_opencl) / 1000;
        std::cout << "\n        ,\"Total\": " << ((float) time_total) / 1000;
        std::cout << "\n    }";
        std::cout << "\n}\n";
        std::cout.flush();
        if (sols > 0) {
            exit(10);
        } else {
            exit(20);
        }
    }
    catch (cl::Error &err) {
        std::cerr << "\nERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
        if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
            std::string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cout << "Program Info: " << str << std::endl;
        }
    }
    catch (std::string &msg) {
        std::cerr << "Exception caught in main(): " << msg << std::endl;
    }
}

bool isPrimalGraph(satformulaType *satFormula, treedecType *treeDecomp) {
    for (auto &clause : satFormula->clauses) {
        for (int x = 0; x < clause.size(); x++) {
            for (int y = x + 1; y < clause.size(); y++) {
                bool found = false;
                for (int j = 0; j < treeDecomp->numb; j++) {
                    if (std::find(treeDecomp->bags[j].variables.begin(), treeDecomp->bags[j].variables.end(), std::abs(clause[x])) != treeDecomp->bags[j].variables.end() && std::find(treeDecomp->bags[j].variables.begin(), treeDecomp->bags[j].variables.end(), std::abs(clause[y])) != treeDecomp->bags[j].variables.end()) {
                        found = true;
                    }
                }
                if (!found) {
                    return false;
                }
            }
        }
    }
    return true;
}

long long int getTime() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}