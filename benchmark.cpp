#include <stdio.h>
#include <time.h>
#include <cstdint>
#include <format>
#include <fstream>
#include <iostream>
#include <print>
#include <ranges>
#include <set>

#include <cubool.h>

#include <fast_matrix_market/fast_matrix_market.hpp>

#include "regular_path_query.hpp"
#include "timer.hpp"

#define LEN 512
#define MAX_LABELS 16
#define RUNS 1

#define WIKIDATA_DIR "/home/mitya/Wikidata/"
#define QUERIES_DIR "/home/mitya/Queries/"
#define RESULTS_DIR "Results/"

#define QUERY_COUNT 660
#define PROP_COUNT 1395

struct MatrixData {
  bool loaded = false;
  int64_t nrows = 0, ncols = 0;
  std::vector<cuBool_Index> rows, cols;
  cuBool_Index nvals = 0;

  cuBool_Matrix matrix = nullptr;

  double sizeMb() { return (sizeof(cuBool_Index) * nvals * 2) / 1'000'000.0; }
};

using Wikidata = std::vector<MatrixData>;

struct Query {
  std::vector<cuBool_Matrix> graph;
  std::vector<cuBool_Matrix> automat;

  std::vector<cuBool_Index> sourece_vertices;
  std::vector<cuBool_Index> start_states;

  std::vector<cuBool_Index> final_states;

  std::vector<uint32_t> labels;
  std::vector<bool> inverse_lables;
  bool labels_inversed = false;

  bool matrices_was_loaded = true;
};

static bool load_matrix(MatrixData &data, std::string_view filename) {
  if (data.loaded) {
    return true;
  }

  std::ifstream file(filename.data());
  if (not file) {
    return false;
  }

  std::vector<bool> vals;
  fast_matrix_market::read_matrix_market_triplet(file, data.nrows, data.ncols, data.rows, data.cols,
                                                 vals);
  data.nvals = vals.size();
  data.loaded = true;

  return true;
}

static bool create_matrix(cuBool_Matrix *matrix, const MatrixData &data) {
  cuBool_Status status = CUBOOL_STATUS_SUCCESS;

  status = cuBool_Matrix_New(matrix, data.nrows, data.ncols);
  if (status != CUBOOL_STATUS_SUCCESS) {
    return false;
  }

  status =
    cuBool_Matrix_Build(*matrix, data.rows.data(), data.cols.data(), data.nvals, CUBOOL_HINT_NO);
  if (status != CUBOOL_STATUS_SUCCESS) {
    return false;
  }

  return true;
}

static Wikidata load_matrices(bool load_at_gpu = false) {
  Wikidata matrices(PROP_COUNT + 1);

  Timer::mark();
  std::cout << "loading at RAM\n";
  for (uint32_t query_number = 0; query_number < QUERY_COUNT; query_number++) {
    std::cout << "\rloaded query # " << query_number;
    std::flush(std::cout);

    std::string filename = std::format("{}{}/meta.txt", QUERIES_DIR, query_number);
    std::ifstream query_file(filename);
    if (not query_file) {
      continue;
    }

    // read sourse and dest
    int tmp1, tmp2;
    query_file >> tmp1 >> tmp2;
    if (tmp1 == 0 && tmp2 == 0) {
      continue;
    }

    // read start vertices and start states
    for (int _ = 0; _ < 2; _++) {
      query_file >> tmp1;
      for (int i = 0; i < tmp1; i++) {
        query_file >> tmp2;
      }
    }

    uint32_t labels_number = 0;
    query_file >> labels_number;
    for (int i = 0; i < labels_number; i++) {
      int label;
      query_file >> label;
      label = std::abs(label);

      std::string filename = std::format("{}{}.txt", WIKIDATA_DIR, label);
      load_matrix(matrices[label], filename);
    }
  }
  std::cout << "\r";
  double elapsed = Timer::measure();
  std::cout << "matrices loaded, time: " << elapsed << "s\n";

  if (load_at_gpu) {
    Timer::mark();
    std::cout << "loading at VRAM\n";
    for (int i = 0; i < matrices.size(); i++) {
      uint32_t free_mem = parse_int(exec("python3 parse_mem.py"));

      auto &data = matrices[i];
      if (!data.loaded) {
        continue;
      }

      create_matrix(&data.matrix, data);

      uint32_t new_free_mem = parse_int(exec("python3 parse_mem.py"));
      std::println("query #{}: now used: {}, diff used: {}, actual size: {}", i, new_free_mem,
                   new_free_mem - free_mem, data.sizeMb());
    }
    elapsed = Timer::measure();
    std::cout << "matrices loaded at GPU, time: " << elapsed << "s\n";
  }

  return matrices;
}

static bool load_query(Query &query, uint32_t query_number, const Wikidata &matrices,
                       bool preloaded) {
  std::string filename = std::format("{}{}/meta.txt", QUERIES_DIR, query_number);
  std::ifstream query_file(filename);
  if (!query_file) {
    std::cout << "skipped query " << query_number << ", file not exists\n";
    return false;
  }

  cuBool_Index source = 0, dest = 0;
  query_file >> source >> dest;
  if (source == 0 && dest == 0) {
    std::cout << "skipped query " << query_number << ", sourse and dest is 0\n";
    return false;
  }
  source--;
  dest--;

  uint32_t src_verts_number = 0;
  query_file >> src_verts_number;
  std::vector<cuBool_Index> src_verts(src_verts_number);
  for (auto &vert : src_verts) {
    query_file >> vert;
    vert--;
  }

  uint32_t inv_src_vert_number = 0;
  query_file >> inv_src_vert_number;
  std::vector<cuBool_Index> inv_src_verts(src_verts_number);
  for (auto &vert : inv_src_verts) {
    query_file >> vert;
    vert--;
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

    // std::string filename = std::format("{}{}.txt", WIKIDATA_DIR, label);
    if (!preloaded) {
      query.matrices_was_loaded = true;
      if (not create_matrix(&query.graph[i], matrices[label])) {
        return false;
      }
    } else {
      query.matrices_was_loaded = false;
      query.graph[i] = matrices[i].matrix;
    }

    filename = std::format("{}{}/{}.txt", QUERIES_DIR, query_number,
                           query.inverse_lables[i] ? -(int)label : (int)label);
    MatrixData data;
    if (not load_matrix(data, filename) || not create_matrix(&query.automat[i], data)) {
      return false;
    }
  }

  if (source == std::numeric_limits<cuBool_Index>::max()) {
    query.start_states = std::move(inv_src_verts);
    query.final_states = std::move(src_verts);
    query.sourece_vertices = std::vector {dest};
    query.labels_inversed = true;
  } else {
    query.start_states = std::move(src_verts);
    query.final_states = std::move(inv_src_verts);
    query.sourece_vertices = std::vector {source};
    query.labels_inversed = false;
  }

  return true;
}

static void clear_query(Query &query) {
  if (query.matrices_was_loaded) {
    for (auto matrix : query.graph) {
      if (matrix != nullptr) {
        cuBool_Matrix_Free(matrix);
      }
    }
  }

  for (auto matrix : query.automat) {
    if (matrix != nullptr) {
      cuBool_Matrix_Free(matrix);
    }
  }
}

static uint32_t make_query(const Query &query) {
  auto recheable =
    regular_path_query(query.graph, query.sourece_vertices, query.automat, query.start_states,
                       query.inverse_lables, query.labels_inversed);
  cuBool_Vector P, F;

  cuBool_Index automat_rows, graph_rows;
  cuBool_Matrix_Nrows(query.graph.front(), &graph_rows);
  cuBool_Matrix_Nrows(query.automat.front(), &automat_rows);

  cuBool_Vector_New(&P, graph_rows);
  cuBool_Vector_New(&F, automat_rows);

  cuBool_Vector_Build(F, query.final_states.data(), query.final_states.size(), CUBOOL_HINT_NO);
  cuBool_VxM(P, F, recheable, CUBOOL_HINT_NO);
  uint32_t answer = 0;
  cuBool_Vector_Nvals(P, &answer);

  cuBool_Vector_Free(P);
  cuBool_Vector_Free(F);
  cuBool_Matrix_Free(recheable);

  return answer;
}

bool benchmark() {
  cuBool_Initialize(CUBOOL_HINT_NO);

  bool preloading = false;
  auto matrices = load_matrices(preloading);
  // save_to_bin(matrices);
  // return true;
  // auto matrices = load_from_bin();

  std::set<uint32_t> too_big_queris = {115};

  double total_load_time = 0;
  double total_execute_time = 0;
  double total_clear_time = 0;

  std::fstream _results_file("result.txt", std::ofstream::out);
  _results_file.close();

  // for (uint32_t query_number = 1; query_number <= QUERY_COUNT; query_number++) {
  for (uint32_t query_number = 1; query_number < 521; query_number++) {
    Query query;

    std::fstream results_file("result.txt", std::ofstream::out | std::ofstream::app);
    if (too_big_queris.contains(query_number)) {
      continue;
    }

    Timer::mark();
    bool status = load_query(query, query_number, matrices, preloading);
    if (!status) {
      std::println("query #{} skipped", query_number);
      clear_query(query);
      continue;
    }
    double load_time = Timer::measure();

    Timer::mark();
    auto result = make_query(query);
    results_file << query_number << ' ' << result << '\n';
    double execute_time = Timer::measure();

    Timer::mark();
    clear_query(query);
    double clear_time = Timer::measure();

    std::println("query #{}; load time: {}, execute time: {}, clear time: {}, result: {}",
                 query_number, load_time, execute_time, clear_time, result);

    total_load_time += load_time;
    total_execute_time += execute_time;
    total_clear_time += clear_time;

    std::println("free space after #3: {}", parse_int(exec("python3 ../parse_mem.py")));
  }

  std::cout << "\n\n\n";
  std::println("total load time: {}, total execute time: {}, total clear time: {}\n",
               total_load_time, total_execute_time, total_clear_time);

  cuBool_Finalize();

  return true;
}
