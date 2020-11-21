#define GPU_HOST_ATTR

#include <sstream>
#include <htd/TreeDecompositionOptimizationOperation.hpp>
#include <htd/ITreeDecompositionAlgorithm.hpp>
#include <htd/BucketEliminationTreeDecompositionAlgorithm.hpp>
#include <htd/IterativeImprovementTreeDecompositionAlgorithm.hpp>
#include <htd/GraphPreprocessor.hpp>
#include <vector>

#include "decomposer.h"

namespace gpusat {

    // Inspired from htd::TdFormatExporter
    treedecType Decomposer::htd_to_bags(const htd::ITreeDecomposition& decomposition, const struct satformulaType& formula) {
        // pairs of (bag, child indices)

        auto decomp = treedecType();
        decomp.numb = decomposition.vertexCount();
        decomp.width = decomposition.maximumBagSize();

        auto not_fact = [&](htd::vertex_t variable) -> bool {
            //return true;
            return !std::binary_search(
                    formula.facts.begin(),
                    formula.facts.end(),
                    variable,
                    compVars
            );
        };

        if (decomposition.vertexCount() > 0) {
            std::vector<htd::vertex_t> vertex_stack;
            std::vector<BagType*> parent_stack;

            {
                auto bag = BagType();
                bag.id = decomposition.root() - 1;
                auto bag_content = decomposition.bagContent(decomposition.root());
                std::copy_if(bag_content.begin(), bag_content.end(), std::back_inserter(bag.variables), not_fact);
                std::sort(bag.variables.begin(), bag.variables.end());

                auto children = decomposition.children(decomposition.root());
                vertex_stack.push_back(0ul);
                vertex_stack.resize(vertex_stack.size() + children.size());
                for (int i=0; i < children.size(); i++) {
                    vertex_stack[vertex_stack.size() - 1 - i] = children[i];
                }
                decomp.root = std::move(bag);
                parent_stack.push_back(&decomp.root);
            }

            while (!vertex_stack.empty()) {
                auto root = vertex_stack.back();
                vertex_stack.pop_back();
                // one child layer finished
                if (root == 0ul) {
                    parent_stack.pop_back();
                    continue;
                }
                auto bag = BagType();
                bag.id = root - 1;
                auto bag_content = decomposition.bagContent(root);
                std::copy_if(bag_content.begin(), bag_content.end(), std::back_inserter(bag.variables), not_fact);
                std::sort(bag.variables.begin(), bag.variables.end());
                parent_stack.back()->edges.push_back(std::move(bag));
                auto children = decomposition.children(root);
                if (!children.empty()) {
                    vertex_stack.push_back(0ul);
                    vertex_stack.resize(vertex_stack.size() + children.size());
                    for (int i=0; i < children.size(); i++) {
                        vertex_stack[vertex_stack.size() - 1 - i] = children[i];
                    }
                    parent_stack.push_back(&parent_stack.back()->edges.back());
                }
            }
            return std::move(decomp);
        }
        return treedecType();
    }

    void Decomposer::gpusat_formula_to_hypergraph(htd::Hypergraph& hypergraph, const satformulaType& formula) {

        hypergraph.addVertices(formula.numVars);

        // Facts should be ignor-able, but they are added in the original
        /*
        for (auto fact : formula.facts) {
            std::vector<htd::vertex_t> clause;
            clause.push_back(htd::vertex_t(std::abs(fact)));
            hypergraph.addEdge(clause);
        }
        */

        auto it = formula.clause_bag.begin();
        while (it != formula.clause_bag.end()) {
            std::vector<htd::vertex_t> clause;
            while (*it != 0) {
                clause.push_back(htd::vertex_t(std::abs(*it)));
                it++;
            }
            hypergraph.addEdge(clause);
            it++;
        }
    }

    treedecType Decomposer::computeDecomposition(const satformulaType& formula, htd::ITreeDecompositionFitnessFunction *fitness, size_t n) {

        htd::LibraryInstance *htdManager = htd::createManagementInstance(htd::Id::FIRST);
        htd::Hypergraph hypergraph(htdManager);

        gpusat_formula_to_hypergraph(hypergraph, formula);
        std::cout << "parsed." << std::endl;

        htd::BucketEliminationTreeDecompositionAlgorithm *treeDecompositionAlgorithm = new htd::BucketEliminationTreeDecompositionAlgorithm(htdManager);
        htd::IterativeImprovementTreeDecompositionAlgorithm *algorithm = new htd::IterativeImprovementTreeDecompositionAlgorithm(htdManager, treeDecompositionAlgorithm, fitness);
        algorithm->setIterationCount(n);
        htd::ITreeDecomposition *decomp = algorithm->computeDecomposition(hypergraph);
        std::cout << "Decomposition Width: " << decomp->maximumBagSize() << std::endl;
        auto gpusat_decomp = htd_to_bags(*decomp, formula);
        gpusat_decomp.numVars = formula.numVars;
        delete algorithm;
        delete decomp;
        delete htdManager;
        return gpusat_decomp;
    }
}
