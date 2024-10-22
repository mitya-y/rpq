#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include <cubool.h>

cuBool_Matrix regular_path_query(
    // vector of sparse graph matrices for each label
    const std::vector<cuBool_Matrix> &graph,
    const std::vector<cuBool_Index> &source_vertices,

    // vector of sparse automat matrices for each label
    const std::vector<cuBool_Matrix> &automat,
    const std::vector<cuBool_Index> &start_states
    ) {
  // this is pointers to normal matrix.
  cuBool_Matrix frontier {}, symbol_frontier {}, next_frontier {}; 

  cuBool_Index graph_nodes_number = 0;
  cuBool_Index automat_nodes_number = 0;

  // transpose graph matrices
  std::vector<cuBool_Matrix> graph_transpsed;
  graph_transpsed.reserve(graph.size());
  for (auto label_matrix : graph) {
    graph_transpsed.emplace_back();
    cuBool_Matrix_Transpose(graph_transpsed.back(), label_matrix, CUBOOL_HINT_NO);
  }

  // transpose automat matrices
  std::vector<cuBool_Matrix> automat_transpsed;
  automat_transpsed.reserve(graph.size());
  for (auto label_matrix : automat) {
    automat_transpsed.emplace_back();
    cuBool_Matrix_Transpose(automat_transpsed.back(), label_matrix, CUBOOL_HINT_NO);
  }

  // get number of graph nodes
  for (auto label_matrix : graph) {
    if (label_matrix != nullptr) {
      cuBool_Matrix_Nrows(label_matrix, &graph_nodes_number);
      break;
    }
  }

  // get number of automat nodes
  for (auto label_matrix : automat) {
    if (label_matrix != nullptr) {
      cuBool_Matrix_Nrows(label_matrix, &automat_nodes_number);
      break;
    }
  }

  // this will be answer
  cuBool_Matrix reacheble {};
  cuBool_Matrix_New(&reacheble, automat_nodes_number, graph_nodes_number);

  // allocate neccessary for algorithm matrices
  cuBool_Matrix_New(&next_frontier, automat_nodes_number, graph_nodes_number);
  cuBool_Matrix_New(&frontier, automat_nodes_number, graph_nodes_number);
  cuBool_Matrix_New(&symbol_frontier, automat_nodes_number, graph_nodes_number);

  // init start values of algorithm matricies
  for (const auto state : start_states) {
    for (const auto vert : source_vertices) {
      assert(vert < graph_nodes_number);
      cuBool_Matrix_SetElement(next_frontier, state, vert);
      cuBool_Matrix_SetElement(reacheble, state, vert);
    }
  }

  cuBool_Index states = source_vertices.size();

  const auto label_number = std::max(graph.size(), automat.size());
  while (states > 0) {
    std::swap(frontier, next_frontier);

    // TODO: clear next_frontier?

    for (int i = 0; i < label_number; i++) {
      if (graph[i] == nullptr || automat[i] == nullptr) {
        continue;
      }
      cuBool_Matrix_EWiseMult(symbol_frontier, automat_transpsed[i], frontier, CUBOOL_HINT_NO);
      // this must be mult with mask: next_frontier = (symbol_frontier * graph[i]) & reacheble
      cuBool_Matrix_EWiseMult(next_frontier, symbol_frontier, graph[i], CUBOOL_HINT_NO);
    }

    // this must be assign with matrix: reacheble = next_frontier & reacheble
    // also maybe here leak old reacheble memory
    cuBool_Matrix_Duplicate(next_frontier, &reacheble);
    cuBool_Matrix_Nvals(next_frontier, &states);
  }

  return reacheble;
}

void test() {
}

int main() {
  std::cout << "hello world\n";
}
