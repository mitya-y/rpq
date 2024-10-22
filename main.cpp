#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <string_view>

#include <cubool.h>

#include <fast_matrix_market/fast_matrix_market.hpp>

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

  cuBool_Matrix_Free(next_frontier);
  cuBool_Matrix_Free(frontier);
  cuBool_Matrix_Free(symbol_frontier);

  return reacheble;
}

void test() {
  cuBool_Initialize(CUBOOL_HINT_NO);

  std::vector<cuBool_Matrix> graph;
  std::vector<cuBool_Index> source_vertices;
  std::vector<cuBool_Matrix> automat;
  std::vector<cuBool_Index> start_states;

  std::vector<std::string> graph_data { "test_data/a.mtx", "test_data/b.mtx" };
  std::vector<std::string> automat_data { "test_data/1_a.mtx", "" };
  std::string_view meta_data("test_data/1_meta.txt");
  std::string_view sources_data("test_data/1_sources.txt");

  int64_t nrows = 0, ncols = 0;
  std::vector<cuBool_Index> rows, cols;
  std::vector<bool> vals;

  // load graph
  graph.reserve(graph_data.size());
  for (const auto &data : graph_data) {
    graph.emplace_back();
    if (data.empty()) {
      continue;
    }

    std::ifstream file(data);
    if (!file) {
      throw std::runtime_error("error open file");
    }
    fast_matrix_market::read_matrix_market_triplet(file, nrows, ncols, rows, cols, vals);

    cuBool_Matrix_New(&graph.back(), nrows, ncols);
    cuBool_Matrix_Build(graph.back(), rows.data(), cols.data(), vals.size(), CUBOOL_HINT_NO);
  }

  // load automat
  automat.reserve(automat_data.size());
  for (const auto &data : automat_data) {
    automat.emplace_back();
    if (data.empty()) {
      continue;
    }

    std::ifstream file(data);
    if (!file) {
      throw std::runtime_error("error open file");
    }
    fast_matrix_market::read_matrix_market_triplet(file, nrows, ncols, rows, cols, vals);

    cuBool_Matrix_New(&automat.back(), nrows, ncols);
    cuBool_Matrix_Build(automat.back(), rows.data(), cols.data(), vals.size(), CUBOOL_HINT_NO);
  }

  // temporary hardcoded
  source_vertices = {0};
  start_states = {0};

  auto answer = regular_path_query(graph, source_vertices, automat, start_states);

  cuBool_Matrix_Free(answer);

  cuBool_Finalize();
}

int main() {
  test();
}
