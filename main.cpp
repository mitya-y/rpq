#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>
#include <string_view>

#include <cubool.h>

#include <fast_matrix_market/fast_matrix_market.hpp>

static void print_cubool_matrix(cuBool_Matrix matrix, std::string name = "") {
    if (name != "") {
        std::cout << name << std::endl;
    }

    cuBool_Index nvals;
    cuBool_Matrix_Nvals(matrix, &nvals);
    std::vector<cuBool_Index> rows(nvals), cols(nvals);
    cuBool_Matrix_ExtractPairs(matrix, rows.data(), cols.data(), &nvals);

    for (int i = 0; i < nvals; i++) {
        printf("(%d, %d)\n", rows[i], cols[i]);
    }
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

  cuBool_Matrix result;
  cuBool_Matrix_New(&result, automat_nodes_number, graph_nodes_number);

  const auto label_number = std::min(graph.size(), automat.size());
  while (states > 0) {
    std::swap(frontier, next_frontier);

    // uint32_t nvals;
    // cuBool_Matrix_Nvals(reacheble, &nvals);
    // std::cout << "nvals = " << nvals << ", states = " << states << std::endl;

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
      cuBool_Matrix_ApplyInverted(result, next_frontier, reacheble, CUBOOL_HINT_NO);
      std::swap(result, next_frontier);

      // cuBool_Matrix_Nvals(symbol_frontier, &nvals);
      // std::cout << "nvals symbol_frontier = " << nvals << std::endl;
      // cuBool_Matrix_Nvals(next_frontier, &nvals);
      // std::cout << "nvals next_frontier = " << nvals << std::endl;
    }

    // this must be accumulate with mask and save old value: reacheble += next_frontier & reacheble
    //
    cuBool_Matrix_EWiseAdd(result, reacheble, next_frontier, CUBOOL_HINT_NO);
    std::swap(result, reacheble);

    cuBool_Matrix_Nvals(next_frontier, &states);
  }

  cuBool_Matrix_Free(next_frontier);
  cuBool_Matrix_Free(frontier);
  cuBool_Matrix_Free(symbol_frontier);
  cuBool_Matrix_Free(result);

  return reacheble;
}

struct Config {
  std::vector<std::string> graph_data;
  std::vector<std::string> automat_data;
  std::vector<uint32_t> sources;
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

  source_vertices = config.sources;
  for (cuBool_Index &s : source_vertices) {
    s--;
  }

  auto answer = regular_path_query(graph, source_vertices, automat, start_states);

  // uint32_t n;
  // cuBool_Matrix_Nvals(answer, &n);
  // std::cout << n << std::endl;

  print_cubool_matrix(answer, "result");
  printf("\n");

  cuBool_Matrix_Free(answer);

  cuBool_Finalize();
}

int main() {
  std::vector<Config> configs {
    {
      .graph_data = { "test_data/example/graph_a.mtx", "test_data/example/graph_b.mtx" },
      .automat_data = { "test_data/example/automat_a.mtx", "test_data/example/automat_b.mtx" },
      .sources = {1},
    },
    {
      .graph_data = { "test_data/a.mtx", "test_data/b.mtx" },
      .automat_data { "test_data/1_a.mtx", "" },
      .sources = {1},
    },
    {
      .graph_data = { "test_data/a.mtx", "test_data/b.mtx" },
      .automat_data { "test_data/2_a.mtx", "test_data/2_b.mtx" },
      .sources = {2},
    },
    {
      .graph_data = { "test_data/a.mtx", "test_data/b.mtx" },
      .automat_data { "test_data/3_a.mtx", "test_data/3_b.mtx" },
      .sources = {3, 6},
    },
    {
      .graph_data = { "test_data/a.mtx", "test_data/b.mtx" },
      .automat_data { "", "test_data/4_b.mtx" },
      .sources = {4},
    },
  };

  // configs.resize(3);
  for (const auto &config : configs) {
    test(config);
  }
}

