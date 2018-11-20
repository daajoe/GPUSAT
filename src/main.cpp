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
#include <decomposer.h>
#include <solver.h>
#include <CLI11.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <gpusatpreprocessor.h>
#include <FitnessFunctions/WidthFitnessFunction.h>
#include <FitnessFunctions/WidthCutSetFitnessFunction.h>
#include <FitnessFunctions/CutSetFitnessFunction.h>
#include <FitnessFunctions/CutSetWidthFitnessFunction.h>

using namespace gpusat;

long long int getTime();

void getKernel(std::vector<cl::Platform> &platforms, cl::Context &context, std::vector<cl::Device> &devices, cl::CommandQueue &queue, cl::Program &program, cl_long &memorySize, cl_long &maxMemoryBuffer, bool nvidia, bool amd, bool cpu, int &combineWidth, std::string kernelPath) {

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
            memorySize = devices[0].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
            maxMemoryBuffer = devices[0].getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
            combineWidth = (int) std::floor(std::log2(devices[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() * devices[0].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()));
            break;
        }
    }
    if (iter == platforms.end()) {
        std::cout << "\nERROR: no GPU found!";
        exit(1);
    }

    struct stat buffer;
    std::string binPath = kernelPath + "SAT_d_p.clbin";

#ifndef DEBUG
    if (stat(binPath.c_str(), &buffer) != 0) {
#endif
    //create kernel binary if it doesn't exist
    std::string sourcePath;

    sourcePath = kernelPath + "SAT_d_primal.cl";
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
}

int main(int argc, char *argv[]) {
    long long int time_total = getTime();
    std::string inputLine;
    std::string formulaDir;
    std::string fitness;
    std::string decompDir;
    int combineWidth = 10;
    time_t seed = time(0);
    std::string kernelPath = "./kernel/";
    bool cpu, weighted, nvidia, amd;
    CLI::App app{};
    std::size_t numDecomps = 30;

    std::string filename = "default";
    app.add_option("-s,--seed", seed, "path to the file containing the sat formula")->set_default_str("");
    app.add_option("-f,--formula", formulaDir, "path to the file containing the sat formula")->set_default_str("");
    app.add_option("-d,--decomposition", decompDir, "path to the file containing the tree decomposition")->set_default_str("");
    app.add_option("-c,--kernelDir", kernelPath, "directory containing the kernel files")->set_default_str("./kernel/");
    app.add_option("-n,--numDecomps", numDecomps, "");
    app.add_flag("--CPU", cpu, "run the solver on the cpu");
    app.add_flag("--NVIDIA", nvidia, "run the solver on an NVIDIA device");
    app.add_flag("--AMD", amd, "run the solver on an AMD device");
    app.add_flag("--weighted", weighted, "use weighted model count");
    app.add_set("--fitnessFunction", fitness, {"width", "cutSet", "width_cutSet", "cutSet_width"}, "fitness function")->set_default_str("width_cutSet");


    CLI11_PARSE(app, argc, argv)

    srand(seed);

    satformulaType satFormula;
    treedecType treeDecomp;
    CNFParser cnfParser(weighted);
    TDParser tdParser(combineWidth);
    {
        std::stringbuf treeD, sat;
        if (formulaDir != "") {
            std::ifstream fileIn(formulaDir);
            while (getline(fileIn, inputLine)) {
                sat.sputn(inputLine.c_str(), inputLine.size());
                sat.sputn("\n", 1);
            }
        } else {
            while (getline(std::cin, inputLine)) {
                sat.sputn(inputLine.c_str(), inputLine.size());
                sat.sputn("\n", 1);
            }
        }

        std::string treeDString;
        if (decompDir != "") {
            std::ifstream fileIn(decompDir);
            while (getline(fileIn, inputLine)) {
                treeD.sputn(inputLine.c_str(), inputLine.size());
                treeD.sputn("\n", 1);
            }
            treeDString = treeD.str();
        } else {
            htd::ITreeDecompositionFitnessFunction *fit;
            if (fitness == "width") {
                fit = new WidthFitnessFunction();
            } else if (fitness == "cutSet") {
                fit = new CutSetFitnessFunction();
            } else if (fitness == "cutSet_width") {
                fit = new CutSetWidthFitnessFunction();
            } else {
                fit = new WidthCutSetFitnessFunction();
            }
            treeDString = Decomposer::computeDecomposition(sat.str(), fit, numDecomps);
        }

        std::string satString = sat.str();

        if (satString.size() < 8) {
            std::cerr << "Error: SAT formula\n";
            exit(EXIT_FAILURE);
        }
        if (treeDString.size() < 8) {
            std::cerr << "Error: tree decomposition\n";
            exit(EXIT_FAILURE);
        }
        satFormula = cnfParser.parseSatFormula(sat.str());
        treeDecomp = tdParser.parseTreeDecomp(treeDString, satFormula);
    }
    std::cout << "\n{\n";
    std::cout << "    \"pre Width\": " << tdParser.preWidth;
    std::cout << "\n    ,\"pre Cut Set Size\": " << tdParser.preCut;
    std::cout << "\n    ,\"pre Join Size\": " << tdParser.preJoinSize;
    std::cout << "\n    ,\"pre Bags\": " << tdParser.preNumBags;
    std::cout.flush();

    Preprocessor::preprocessFacts(treeDecomp, satFormula, tdParser.defaultWeight);
    if (satFormula.unsat) {
        time_total = getTime() - time_total;
        std::cout << "\n    ,\"Model Count\": " << 0;
        std::cout << "\n    ,\"Time\":{";
        std::cout << "\n        \"Solving\": " << 0;
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

    std::vector<cl::Platform> platforms;
    cl::Context context;
    std::vector<cl::Device> devices;
    cl::CommandQueue queue;
    cl::Program program;
    cl_long memorySize = 0;
    cl_long maxMemoryBuffer = 0;

    try {
        getKernel(platforms, context, devices, queue, program, memorySize, maxMemoryBuffer, nvidia, amd, cpu, combineWidth, kernelPath);

        // combine small bags
        Preprocessor::preprocessDecomp(&treeDecomp.bags[0], combineWidth);

        tdParser.iterateDecompPost(treeDecomp.bags[0]);
        tdParser.postNumBags = treeDecomp.bags.size();

        std::cout << "\n    ,\"post Width\": " << tdParser.postWidth;
        std::cout << "\n    ,\"post Cut Set Size\": " << tdParser.postCut;
        std::cout << "\n    ,\"post Join Size\": " << tdParser.postJoinSize;
        std::cout << "\n    ,\"post Bags\": " << tdParser.postNumBags;
        std::cout.flush();

        Solver *sol;
        bagType next;
        sol = new Solver_Primal(context, queue, program, memorySize, maxMemoryBuffer);
        next.variables.assign(treeDecomp.bags[0].variables.begin(), treeDecomp.bags[0].variables.begin() + std::min((cl_long) treeDecomp.bags[0].variables.size(), (cl_long) 12));
        long long int time_solving = getTime();
        (*sol).solveProblem(treeDecomp, satFormula, treeDecomp.bags[0], next, INTRODUCEFORGET);
        time_solving = getTime() - time_solving;

        std::cout << "\n    ,\"Num Join\": " << sol->numJoin;
        std::cout << "\n    ,\"Num Introduce Forget\": " << sol->numIntroduceForget;
        std::cout << "\n    ,\"max Table Size\": " << sol->maxTableSize;

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

            boost::multiprecision::cpp_bin_float_100 base = 2, exponent = treeDecomp.bags[0].correction;
            sols = sols * pow(base, exponent);

            if (weighted) {
                sols = sols * tdParser.defaultWeight;
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