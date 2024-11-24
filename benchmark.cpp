#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <format>
#include <ranges>
#include <set>

#include <cuda.h>
#include <thrust/system_error.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <cuda_profiler_api.h>

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

  double sizeMb() {
    return (sizeof(cuBool_Index) * nvals * 2) / 1'000'000.0;
  }
};
using Wikidata = std::vector<MatrixData>;

struct Query {
  std::vector<cuBool_Matrix> graph;
  std::vector<cuBool_Matrix> automat;

  std::vector<cuBool_Index> sourece_vertices;
  std::vector<cuBool_Index> start_states;

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
  fast_matrix_market::read_matrix_market_triplet(file, data.nrows, data.ncols, data.rows, data.cols, vals);
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

  status = cuBool_Matrix_Build(*matrix, data.rows.data(), data.cols.data(), data.nvals, CUBOOL_HINT_NO);
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
      // std::println(std::cout, "query #{}: now used: {}, diff used: {}, actual size: {}",
      //     i, new_free_mem, new_free_mem - free_mem, data.sizeMb());
      std::cout << std::format("query #{}: now used: {}, diff used: {}, actual size: {}",
          i, new_free_mem, new_free_mem - free_mem, data.sizeMb());
    }
    elapsed = Timer::measure();
    std::cout << "matrices loaded at GPU, time: " << elapsed << "s\n";
  }

  return matrices;
}

static bool load_query(Query &query, uint32_t query_number, const Wikidata &matrices, bool preloaded) {
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
    query.sourece_vertices = std::vector {dest};
    query.labels_inversed = true;
  } else {
    query.start_states = std::move(src_verts);
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

static void make_query(const Query &query) {
  auto recheable = regular_path_query(query.graph, query.sourece_vertices,
                                      query.automat, query.start_states,
                                      query.inverse_lables, query.labels_inversed);
  cuBool_Matrix_Free(recheable);
}

static void save_to_bin(const Wikidata &wiki) {
  std::ofstream f("save.bin", std::ios::binary);
  size_t size = wiki.size();
  f.write(reinterpret_cast<const char *>(&size), sizeof(size_t));
  for (const auto &data : wiki) {
    size = data.cols.size();
    f.write(reinterpret_cast<const char *>(&size), sizeof(size_t));
    f.write(reinterpret_cast<const char *>(data.cols.data()), size * sizeof(cuBool_Index));

    size = data.rows.size();
    f.write(reinterpret_cast<const char *>(&size), sizeof(size_t));
    f.write(reinterpret_cast<const char *>(data.rows.data()), size * sizeof(cuBool_Index));

    f.write(reinterpret_cast<const char *>(&data.loaded), sizeof(bool));

    f.write(reinterpret_cast<const char *>(&data.ncols), sizeof(cuBool_Index));
    f.write(reinterpret_cast<const char *>(&data.nrows), sizeof(cuBool_Index));
    f.write(reinterpret_cast<const char *>(&data.nvals), sizeof(cuBool_Index));
  }
}

static Wikidata load_from_bin() {
  std::ifstream f("save.bin", std::ios::binary);
  size_t size;
  f.read(reinterpret_cast<char *>(&size), sizeof(size_t));
  Wikidata wiki(size);
  for (auto &data : wiki) {
    f.read(reinterpret_cast<char *>(&size), sizeof(size_t));
    data.cols.resize(size);
    f.read(reinterpret_cast<char *>(data.cols.data()), size * sizeof(cuBool_Index));

    f.read(reinterpret_cast<char *>(&size), sizeof(size_t));
    data.rows.resize(size);
    f.read(reinterpret_cast<char *>(data.rows.data()), size * sizeof(cuBool_Index));

    f.read(reinterpret_cast<char *>(&data.loaded), sizeof(bool));

    f.read(reinterpret_cast<char *>(&data.ncols), sizeof(cuBool_Index));
    f.read(reinterpret_cast<char *>(&data.nrows), sizeof(cuBool_Index));
    f.read(reinterpret_cast<char *>(&data.nvals), sizeof(cuBool_Index));
  }

  return wiki;
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

  for (uint32_t query_number = 1; query_number <= QUERY_COUNT; query_number++) {
    Query query;

    if (too_big_queris.contains(query_number)) {
      continue;
    }

    Timer::mark();
    bool status = load_query(query, query_number, matrices, preloading);
    if (!status) {
      std::cout << std::format("query #{} skipped\n", query_number);
      clear_query(query);
      continue;
    }
    double load_time = Timer::measure();

    Timer::mark();
    make_query(query);
    double execute_time = Timer::measure();

    Timer::mark();
    clear_query(query);
    double clear_time = Timer::measure();

    // std::println(std::cout, "query #{}; load time: {}, execute time: {}, clear time: {}",
    //   query_number, load_time, execute_time, clear_time);
    std::cout << std::format("query #{}; load time: {}, execute time: {}, clear time: {}\n",
      query_number, load_time, execute_time, clear_time);

    total_load_time += load_time;
    total_execute_time += execute_time;
    total_clear_time += clear_time;

    // std::cout << std::format("free space after #3: {}\n", parse_int(exec("python3 parse_mem.py")));
  }

  std::cout << "\n\n\n";
  std::cout << std::format("total load time: {}, total execute time: {}, total clear time: {}\n",
    total_load_time, total_execute_time, total_clear_time);

  cuBool_Finalize();

  return true;
}

