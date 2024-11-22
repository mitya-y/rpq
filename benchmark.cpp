#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <format>
#include <ranges>

#include <cubool.h>

#include <fast_matrix_market/fast_matrix_market.hpp>

#include "regular_path_query.hpp"

#define LEN 512
#define MAX_LABELS 16
#define RUNS 1

#define WIKIDATA_DIR "/home/mitya/Wikidata/"
#define QUERIES_DIR "/home/mitya/Queries/"
#define RESULTS_DIR "Results/"

#define QUERY_COUNT 660
#define PROP_COUNT 1395

struct Query {
  cuBool_Index source = 0, dest = 0;

  std::vector<cuBool_Matrix> graph;
  std::vector<cuBool_Matrix> automat;

  std::vector<uint32_t> labels;
  std::vector<bool> inverse_lables;

  std::vector<cuBool_Index> sourece_vertices;
  std::vector<cuBool_Index> start_states;
};

static bool load_matrix(cuBool_Matrix *matrix, std::string_view filename) {
  std::ifstream file(filename.data());
  if (not file) {
    return false;
  }

  int64_t nrows = 0, ncols = 0;
  std::vector<cuBool_Index> rows, cols;
  std::vector<bool> vals;
  fast_matrix_market::read_matrix_market_triplet(file, nrows, ncols, rows, cols, vals);

  cuBool_Matrix_New(matrix, nrows, ncols);
  cuBool_Matrix_Build(*matrix, rows.data(), cols.data(), vals.size(), CUBOOL_HINT_NO);

  return true;
}

static bool load_query(Query &query, uint32_t query_number) {
  std::string filename = std::format("{}{}/meta.txt", QUERIES_DIR, query_number);
  std::ifstream query_file(filename);
  if (not query_file) {
    std::cout << "skipped query " << query_number << ", file not exists\n";
    return false;
  }

  query_file >> query.source >> query.dest;
  if (query.source == 0 && query.dest == 0) {
    std::cout << "skipped query " << query_number << ", sourse and dest is 0\n";
    return false;
  }
  query.source--;
  query.dest--;

  uint32_t src_verts_number = 0;
  query_file >> src_verts_number;
  query.sourece_vertices.resize(src_verts_number);
  for (auto &vert : query.sourece_vertices) {
    query_file >> vert;
    vert--;
  }

  uint32_t start_states_number = 0;
  query_file >> start_states_number;
  query.start_states.resize(start_states_number);
  for (auto &state : query.start_states) {
    query_file >> state;
    state--;
  }

  uint32_t labels_number = 0;
  query_file >> labels_number;
  query.labels.resize(labels_number);
  query.inverse_lables.resize(labels_number);
  for (int i = 0; i < labels_number; i++) {
    int label;
    query_file >> label;
    query.labels[i] = std::abs(label);
    query.inverse_lables[i] = label < 0;
  }

  query.graph.assign(labels_number, nullptr);
  query.automat.assign(labels_number, nullptr);

  for (int i = 0; i < labels_number; i++) {
    uint32_t label = query.labels[i];

    std::string filename = std::format("{}{}.txt", WIKIDATA_DIR, label);
    if (not load_matrix(&query.graph[i], filename)) {
      throw std::runtime_error(("can not find matrix for label " +
                                std::to_string(label)).c_str());
      return false;
    }

    filename = std::format("{}{}/{}.txt", QUERIES_DIR, query_number,
      query.inverse_lables[i] ? -(int)label : (int)label);
    if (not load_matrix(&query.automat[i], filename)) {
      throw std::runtime_error(("can not find matrix for label " +
                                std::to_string(label)).c_str());
      return false;
    }
  }

  return true;
}

static void clear_query(Query &query) {
  for (auto matrix : query.graph) {
    cuBool_Matrix_Free(matrix);
  }

  for (auto matrix : query.automat) {
    cuBool_Matrix_Free(matrix);
  }
}

static void make_query(const Query &query) {
  auto recheable = regular_path_query(query.graph, query.sourece_vertices,
                                      query.automat, query.start_states);

  cuBool_Matrix_Free(recheable);
}


bool benchmark() {
  cuBool_Initialize(CUBOOL_HINT_NO);

  struct timespec start, finish;

  for (uint32_t query_number = 1; query_number <= QUERY_COUNT; query_number++) {
    std::cout << "query number " << query_number << std::endl;

    Query query;
    load_query(query, query_number);
    clear_query(query);
  }

  return true;
}

