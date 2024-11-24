#include <fstream>
#include <iostream>
#include <string_view>

#include <fast_matrix_market/fast_matrix_market.hpp>

#include "regular_path_query.hpp"

struct Config {
  std::vector<std::string> graph_data;
  std::vector<std::string> automat_data;
  std::vector<uint32_t> sources;
  std::string_view meta_data;
  std::string_view sources_data;
};

void test_on_config(const Config &config) {
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

  for (auto matrix : graph) {
    if (matrix != nullptr) {
      cuBool_Matrix_Free(matrix);
    }
  }
  for (auto matrix : automat) {
    if (matrix != nullptr) {
      cuBool_Matrix_Free(matrix);
    }
  }

  cuBool_Finalize();
}

void test() {
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
    test_on_config(config);
  }
}

int main() {
  // test();
  benchmark();
}

