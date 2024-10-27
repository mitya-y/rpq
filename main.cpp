#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <string_view>

#include <cubool.h>

#include <fast_matrix_market/fast_matrix_market.hpp>

// bad temporary algorithm
static void invert_matrix(cuBool_Matrix mask) {
  // invert mask
  cuBool_Index nvals;
  cuBool_Matrix_Nvals(mask, &nvals);

  std::vector<cuBool_Index> rows(nvals), cols(nvals);
  cuBool_Matrix_ExtractPairs(mask, rows.data(), cols.data(), &nvals);

  cuBool_Index ncols, nrows;
  cuBool_Matrix_Ncols(mask, &ncols);
  cuBool_Matrix_Nrows(mask, &nrows);
  std::vector inverted_mask(nrows, std::vector(ncols, true));

  for (int i = 0; i < nvals; i++) {
    inverted_mask[rows[i]][cols[i]] = false;
  }

  rows.clear();
  rows.reserve(ncols * nvals - nvals);
  cols.clear();
  cols.reserve(ncols * nvals - nvals);
  for (cuBool_Index i = 0; i < nrows; i++) {
    for (cuBool_Index j = 0; j < ncols; j++) {
      if (inverted_mask[i][j]) {
        rows.push_back(i);
        cols.push_back(j);
      }
    }
  }

  cuBool_Matrix_Build(mask, rows.data(), cols.data(), rows.size(), CUBOOL_HINT_NO);
}

static void apply_not_mask(cuBool_Matrix matrix, cuBool_Matrix mask) {
  cuBool_Matrix inverted_mask;
  cuBool_Matrix_Duplicate(mask, &inverted_mask);
  invert_matrix(inverted_mask);

  cuBool_Matrix tmp_frontier;
  cuBool_Matrix_Duplicate(matrix, &tmp_frontier);

  cuBool_Matrix_EWiseMult(matrix, tmp_frontier, inverted_mask, CUBOOL_HINT_NO);

  cuBool_Matrix_Free(inverted_mask);
  cuBool_Matrix_Free(tmp_frontier);
}

cuBool_Matrix regular_path_query(
    // vector of sparse graph matrices for each label
    const std::vector<cuBool_Matrix> &graph,
    const std::vector<cuBool_Index> &source_vertices,

    // vector of sparse automat matrices for each label
    const std::vector<cuBool_Matrix> &automat,
    const std::vector<cuBool_Index> &start_states
    ) {
  cuBool_Status status;

  // this is pointers to normal matrix.
  cuBool_Matrix frontier {}, symbol_frontier {}, next_frontier {};

  cuBool_Index graph_nodes_number = 0;
  cuBool_Index automat_nodes_number = 0;

  // transpose graph matrices
  std::vector<cuBool_Matrix> graph_transpsed;
  graph_transpsed.reserve(graph.size());
  for (auto label_matrix : graph) {
    // TODO call new
    graph_transpsed.emplace_back();
    cuBool_Matrix_Transpose(graph_transpsed.back(), label_matrix, CUBOOL_HINT_NO);
  }

  // transpose automat matrices
  std::vector<cuBool_Matrix> automat_transpsed;
  automat_transpsed.reserve(graph.size());
  for (auto label_matrix : automat) {
    cuBool_Index nrows, ncols;
    cuBool_Matrix_Nrows(label_matrix, &nrows);
    cuBool_Matrix_Ncols(label_matrix, &ncols);

    automat_transpsed.emplace_back();
    cuBool_Matrix_New(&automat_transpsed.back(), nrows, ncols);
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

  const auto label_number = std::min(graph.size(), automat.size());
  while (states > 0) {
    std::swap(frontier, next_frontier);

    uint32_t nvals;
    cuBool_Matrix_Nvals(reacheble, &nvals);
    std::cout << "nvals = " << nvals << " states = " << states << std::endl;

    // clear next_frontier
    cuBool_Matrix_Build(next_frontier, nullptr, nullptr, 0, CUBOOL_HINT_NO);

    for (int i = 0; i < label_number; i++) {
      if (graph[i] == nullptr || automat[i] == nullptr) {
        continue;
      }

      // is this overwrite symbol_frontier? not :(
      status = cuBool_MxM(symbol_frontier, automat_transpsed[i], frontier, CUBOOL_HINT_NO);
      assert(status == CUBOOL_STATUS_SUCCESS);

      // this must be mult with mask: next_frontier += (symbol_frontier * graph[i]) & (!reachible)
      status = cuBool_MxM(next_frontier, symbol_frontier, graph[i], CUBOOL_HINT_ACCUMULATE);
      assert(status == CUBOOL_STATUS_SUCCESS);

      // apply mask
      apply_not_mask(next_frontier, reacheble);

      cuBool_Matrix_Nvals(symbol_frontier, &nvals);
      std::cout << "nvals symbol_frontier = " << nvals << std::endl;
      cuBool_Matrix_Nvals(next_frontier, &nvals);
      std::cout << "nvals next_frontier = " << nvals << std::endl;
    }

    // this must be accumulate with mask and save old value: reacheble += next_frontier & reacheble
    cuBool_Matrix tmp_reacheble;
    cuBool_Matrix_Duplicate(reacheble, &tmp_reacheble);
    cuBool_Matrix_EWiseAdd(reacheble, tmp_reacheble, next_frontier, CUBOOL_HINT_NO);
    cuBool_Matrix_Free(tmp_reacheble);

    cuBool_Matrix_Nvals(next_frontier, &states);
  }

  cuBool_Matrix_Free(next_frontier);
  cuBool_Matrix_Free(frontier);
  cuBool_Matrix_Free(symbol_frontier);

  return reacheble;
}

struct Config {
  std::vector<std::string> graph_data;
  std::vector<std::string> automat_data;
  std::string_view meta_data;
  std::string_view sources_data;
};

void test(const Config &config) {
  cuBool_Initialize(CUBOOL_HINT_NO);

  std::vector<cuBool_Matrix> graph;
  std::vector<cuBool_Index> source_vertices;
  std::vector<cuBool_Matrix> automat;
  std::vector<cuBool_Index> start_states;

  int64_t nrows = 0, ncols = 0;
  std::vector<cuBool_Index> rows, cols;
  std::vector<bool> vals;

  // load graph
  graph.reserve(config.graph_data.size());
  for (const auto &data : config.graph_data) {
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
  automat.reserve(config.automat_data.size());
  for (const auto &data : config.automat_data) {
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

  uint32_t n;
  cuBool_Matrix_Nvals(answer, &n);
  std::cout << n << std::endl;
  // cuBool_Matrix_Nrows(answer, &n);
  // std::cout << n << std::endl;
  // cuBool_Matrix_Ncols(answer, &n);
  // std::cout << n << std::endl;

  cuBool_Matrix_Free(answer);

  cuBool_Finalize();
}

int main() {
  Config config {
    .graph_data = { "test_data/a.mtx", "test_data/b.mtx" },
    .automat_data { "test_data/1_a.mtx", "" },
    .meta_data = "test_data/1_meta.txt",
    .sources_data = "test_data/1_sources.txt",
  };

  config = Config {
    .graph_data = { "test_data/example/graph_a.mtx", "test_data/example/graph_b.mtx" },
    .automat_data = { "test_data/example/automat_a.mtx", "test_data/example/automat_b.mtx" },
  };

  test(config);
}
