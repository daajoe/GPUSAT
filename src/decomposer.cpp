#include <decomposer.h>
#include <sstream>
#include <htd/TreeDecompositionOptimizationOperation.hpp>
#include <htd/ITreeDecompositionAlgorithm.hpp>
#include <htd/BucketEliminationTreeDecompositionAlgorithm.hpp>
#include <htd/IterativeImprovementTreeDecompositionAlgorithm.hpp>
#include <htd/GraphPreprocessor.hpp>
#include <htd_io/TdFormatExporter.hpp>

namespace gpusat {
    std::string Decomposer::computeDecomposition(std::string formula, htd::ITreeDecompositionFitnessFunction *fitness, size_t n) {

        htd::LibraryInstance *htdManager = htd::createManagementInstance(htd::Id::FIRST);
        htd::Hypergraph hypergraph(htdManager);

        std::stringstream ss(formula);
        std::string item;
        while (getline(ss, item)) {
            //ignore empty line
            if (item.length() > 0) {
                //replace tabs with spaces
				std::replace(item.begin(), item.end(), '\t', ' ');
                char type = item.at(0);
                if (type == 'c' || type == '%') {
                    //comment line (ignore)
                } else if (type == 'p') {
                    //start line
                    parseProblemLine(item, hypergraph);
                } else if (type == 'w' || type == 's') {
                    //weight line (ignore)
                } else if (item.size() > 0) {
                    //clause line
                    parseClauseLine(item, hypergraph);
                }
            }
        }
        htd::BucketEliminationTreeDecompositionAlgorithm *treeDecompositionAlgorithm = new htd::BucketEliminationTreeDecompositionAlgorithm(htdManager);
        htd::IterativeImprovementTreeDecompositionAlgorithm *algorithm = new htd::IterativeImprovementTreeDecompositionAlgorithm(htdManager, treeDecompositionAlgorithm, fitness);
        algorithm->setIterationCount(n);
        htd::ITreeDecomposition *decomp = algorithm->computeDecomposition(hypergraph);
        htd_io::TdFormatExporter exp;
        std::ostringstream oss;
        exp.write(*decomp, hypergraph, oss);
        delete algorithm;
        delete decomp;
        delete htdManager;
        return oss.str();
    }

    void Decomposer::parseProblemLine(std::string line, htd::Hypergraph &hypergraph) {
        std::stringstream sline(line);
        std::string i;
        getline(sline, i, ' '); //p
        while (i.size() == 0) getline(sline, i, ' ');
        getline(sline, i, ' '); //cnf
        while (i.size() == 0) getline(sline, i, ' ');
        getline(sline, i, ' '); //num vars
        while (i.size() == 0) getline(sline, i, ' ');
        hypergraph.addVertices(stoi(i));
    }

    void Decomposer::parseClauseLine(std::string line, htd::Hypergraph &hypergraph) {
        std::vector<htd::vertex_t> clause;
        std::stringstream sline(line);
        std::string i;

        long num = 0;
        while (!sline.eof()) {
            getline(sline, i, ' ');
            if (i.size() > 0) {
                num = stol(i);
                if (num != 0) {
                    clause.push_back(std::abs(num));
                }
            }
        }
        hypergraph.addEdge(clause);
    }
}
