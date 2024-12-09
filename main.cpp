#include <cstdint>
#include <fstream>
#include <iostream>
#include <string_view>
#include <print>
#include <numeric>
#include <set>

#include <fast_matrix_market/fast_matrix_market.hpp>
#include <vector_types.h>

#include "cubool.h"
#include "regular_path_query.hpp"

struct Config {
  std::vector<std::string> graph_data;
  std::vector<std::string> automat_data;
  std::vector<uint32_t> sources;
  std::string_view expexted;
  std::string_view meta_data;
  std::string_view sources_data;
};

bool test_on_config(const Config &config) {
  cuBool_Initialize(CUBOOL_HINT_NO);

  std::vector<cuBool_Matrix> graph;
  std::vector<cuBool_Index> source_vertices;
  std::vector<cuBool_Matrix> automat;
  std::vector<cuBool_Index> start_states;

  int64_t nrows = 0, ncols = 0;
  std::vector<cuBool_Index> rows, cols;
  std::vector<bool> vals;

  // load graph
  cuBool_Index graph_rows = 0;
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

    graph_rows = std::max(graph_rows, (cuBool_Index)nrows);
  }

  cuBool_Index automat_rows = 0;
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

    automat_rows = std::max(automat_rows, (cuBool_Index)nrows);
  }

  // temporary hardcoded
  source_vertices = {0};
  start_states = {0};

  source_vertices = config.sources;
  for (cuBool_Index &s : source_vertices) {
    s--;
  }

  auto answer = regular_path_query(graph, source_vertices, automat, start_states);

  cuBool_Vector P, F;
  cuBool_Vector_New(&P, graph_rows);
  cuBool_Vector_New(&F, automat_rows);

  std::vector<cuBool_Index> final_states(automat_rows);
  std::iota(final_states.begin(), final_states.end(), 0);

  cuBool_Vector_Build(F, final_states.data(), final_states.size(), CUBOOL_HINT_NO);
  cuBool_VxM(P, F, answer, CUBOOL_HINT_NO);
  uint32_t result = 0;
  cuBool_Vector_Nvals(P, &result);

  print_cubool_matrix(answer, "answer");
  print_cubool_vector(F, "F");
  print_cubool_vector(P, "P");
  printf("\n");

  bool test_result = true;

  // validate answer
  cuBool_Index nvals;
  cuBool_Matrix_Nvals(answer, &nvals);
  rows.resize(nvals);
  cols.resize(nvals);
  cuBool_Matrix_ExtractPairs(answer, rows.data(), cols.data(), &nvals);

  std::set<std::pair<cuBool_Index, cuBool_Index>> indexes {};
  for (int i = 0; i < nvals; i++) {
    indexes.insert({rows[i], cols[i]});
  }

  std::ifstream expected_file(config.expexted.data());
  cuBool_Index expected_nvals;
  expected_file >> expected_nvals;

  if (expected_nvals != nvals) {
    test_result = false;
  } else {
    for (int k = 0; k < expected_nvals; k++) {
      int i, j;
      expected_file >> i >> j;
      if (!indexes.contains({i, j})) {
        test_result = false;
        break;
      }
    }
  }

  cuBool_Vector_Free(F);
  cuBool_Vector_Free(P);
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

  return test_result;
}

bool test() {
  std::vector<Config> configs {
    {
      .graph_data = { "test_data/example/graph_a.mtx", "test_data/example/graph_b.mtx" },
      .automat_data = { "test_data/example/automat_a.mtx", "test_data/example/automat_b.mtx" },
      .sources = {1},
      .expexted = "test_data/example/expected.txt",
    },
    {
      .graph_data = { "test_data/a.mtx", "test_data/b.mtx" },
      .automat_data { "test_data/1_a.mtx", "" },
      .sources = {1},
      .expexted = "test_data/1_expected.txt",
    },
    {
      .graph_data = { "test_data/a.mtx", "test_data/b.mtx" },
      .automat_data { "test_data/2_a.mtx", "test_data/2_b.mtx" },
      .sources = {2},
      .expexted = "test_data/2_expected.txt",
    },
    {
      .graph_data = { "test_data/a.mtx", "test_data/b.mtx" },
      .automat_data { "test_data/3_a.mtx", "test_data/3_b.mtx" },
      .sources = {3, 6},
      .expexted = "test_data/3_expected.txt",
    },
    {
      .graph_data = { "test_data/a.mtx", "test_data/b.mtx" },
      .automat_data { "", "test_data/4_b.mtx" },
      .sources = {4},
      .expexted = "test_data/4_expected.txt",
    },
  };

  // configs.resize(1);
  int i = 0;
  for (const auto &config : configs) {
    if (!test_on_config(config)) {
      std::println("Test case {} failed!", i);
      return false;
    }
    i++;
  }
  return true;
}

int main() {
  exit(test() ? 0 : -1);
  // benchmark();
}

