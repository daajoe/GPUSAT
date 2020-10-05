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
    treedecType Decomposer::htd_to_bags(const htd::ITreeDecomposition& decomposition) {
        // pairs of (bag, child indices)
        std::vector<std::pair<BagType, std::vector<size_t>>> bags;
        bags.reserve(decomposition.vertexCount());

        if (decomposition.vertexCount() > 0) {

            for (htd::vertex_t node : decomposition.vertices()) {
                auto bag = BagType();

                bag.id = bags.size();
                for (htd::vertex_t vertex : decomposition.bagContent(node)) {
                    bag.variables.push_back(vertex);
                }

                std::sort(bag.variables.begin(), bag.variables.end());

                bags.push_back(std::pair(std::move(bag), std::vector<size_t>()));
            }

            const auto& hyperedgeCollection = decomposition.hyperedges();

            size_t edge_count = decomposition.edgeCount();

            auto it = hyperedgeCollection.begin();

            for (htd::index_t edge_index = 0; edge_index < edge_count; edge_index++) {
                auto& hyperedge = *it;
                assert(hyperedge.size() == 2);
                bags[hyperedge.at(0) - 1].second.push_back(hyperedge.at(1) - 1);
                it++;
            }


            // indices of the bags which still do not have all children
            std::vector<size_t> incomplete_bag_indices;
            incomplete_bag_indices.reserve(bags.size());
            for (size_t i=0; i < bags.size(); i++) {
                incomplete_bag_indices.push_back(i);
            }

            // FIXME: This could be done more efficiently
            while (!incomplete_bag_indices.empty()) {
                for (auto it = incomplete_bag_indices.cbegin(); it != incomplete_bag_indices.cend(); it++) {
                    auto& bag = bags[*it];
                    for (auto e_it = bag.second.cbegin(); e_it != bag.second.cend();) {
                        bool incomplete = std::binary_search(
                            incomplete_bag_indices.begin(),
                            incomplete_bag_indices.end(),
                            *e_it
                        );
                        // child is done
                        if (!incomplete) {
                            bag.first.edges.push_back(std::move(bags[*e_it].first));
                            e_it = bag.second.erase(e_it);
                        } else {
                            e_it++;
                        }
                    }

                    // no child edges remaining
                    if (bag.second.empty()) {
                        std::sort(bag.first.edges.begin(), bag.first.edges.end(), compTreedType);
                        incomplete_bag_indices.erase(it);
                        break;
                    }
                }
            }
            auto decomp = treedecType();
            decomp.numb = decomposition.vertexCount();
            decomp.width = decomposition.maximumBagSize();
            decomp.root = std::move(bags[0].first);
            return std::move(decomp);
        }
        return treedecType();
    }

    void Decomposer::gpusat_formula_to_hypergraph(htd::Hypergraph& hypergraph, const satformulaType& formula) {

        hypergraph.addVertices(formula.numVars);

        std::vector<htd::vertex_t> clause;
        // Facts should be ignor-able, but they are added in the original
        for (auto fact : formula.facts) {
            clause.push_back(htd::vertex_t(std::abs(fact)));
            hypergraph.addEdge(clause);
            clause.clear();
        }

        for (auto f_clause : formula.clauses) {
            for (auto var : f_clause) {
                clause.push_back(htd::vertex_t(std::abs(var)));
            }
            hypergraph.addEdge(clause);
            clause.clear();
        }
    }

    treedecType Decomposer::computeDecomposition(const satformulaType& formula, htd::ITreeDecompositionFitnessFunction *fitness, size_t n) {

        htd::LibraryInstance *htdManager = htd::createManagementInstance(htd::Id::FIRST);
        htd::Hypergraph hypergraph(htdManager);

        gpusat_formula_to_hypergraph(hypergraph, formula);

        htd::BucketEliminationTreeDecompositionAlgorithm *treeDecompositionAlgorithm = new htd::BucketEliminationTreeDecompositionAlgorithm(htdManager);
        htd::IterativeImprovementTreeDecompositionAlgorithm *algorithm = new htd::IterativeImprovementTreeDecompositionAlgorithm(htdManager, treeDecompositionAlgorithm, fitness);
        algorithm->setIterationCount(n);
        htd::ITreeDecomposition *decomp = algorithm->computeDecomposition(hypergraph);
        std::cout << "Decomposition Width: " << decomp->maximumBagSize() << std::endl;
        auto gpusat_decomp = htd_to_bags(*decomp);
        gpusat_decomp.numVars = formula.numVars;
        delete algorithm;
        delete decomp;
        delete htdManager;
        return gpusat_decomp;
    }
}
