#include <stdio.h>
#include <time.h>
#include <cstdint>
#include <filesystem>
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

#define QUERIES_LOGS "queries_logs"

struct MatrixData {
  bool _loaded = false;
  int64_t _nrows = 0, _ncols = 0;
  std::vector<cuBool_Index> _rows, _cols;
  cuBool_Index _nvals = 0;

  cuBool_Matrix _matrix = nullptr, _transposed = nullptr;

  double sizeMb() const {
    return (sizeof(cuBool_Index) * _nvals * 2) / 1'000'000.0;
  }

  bool load_to_cpu(std::string_view filename);
  bool copy_to_gpu(cuBool_Matrix *matrix) const;

  bool load_to_gpu() {
    return copy_to_gpu(&_matrix);
  }

  ~MatrixData() {
    if (_matrix != nullptr) {
      cuBool_Matrix_Free(_matrix);
    }
  }
};
using Wikidata = std::vector<MatrixData>;

bool MatrixData::load_to_cpu(std::string_view filename) {
  if (_loaded) {
    return true;
  }

  std::ifstream file(filename.data());
  if (not file) {
    return false;
  }

  std::vector<bool> vals;
  fast_matrix_market::read_matrix_market_triplet(file, _nrows, _ncols, _rows, _cols, vals);
  _nvals = vals.size();
  _loaded = true;

  return true;
}

bool MatrixData::copy_to_gpu(cuBool_Matrix *matrix) const {
  cuBool_Status status = CUBOOL_STATUS_SUCCESS;

  status = cuBool_Matrix_New(matrix, _nrows, _ncols);
  if (status != CUBOOL_STATUS_SUCCESS) {
    return false;
  }

  status =
    cuBool_Matrix_Build(*matrix, _rows.data(), _cols.data(), _nvals, CUBOOL_HINT_NO);
  if (status != CUBOOL_STATUS_SUCCESS) {
    return false;
  }

  return true;
}

static Wikidata load_matrices(bool load_at_gpu = false, bool pretransposed = false) {
  Wikidata matrices(BENCH_LABEL_COUNT + 1);
  Timer load_matrices_timer {};

  load_matrices_timer.mark();
  std::cout << "loading at RAM\n";
  for (uint32_t query_number = 1; query_number <= BENCH_QUERY_COUNT; query_number++) {
    std::cout << "\rloaded query # " << query_number;
    std::flush(std::cout);

    std::string filename = std::format("{}{}{}/meta.txt", BENCH_DATASET_DIR, "/Queries/", query_number);
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

      std::string filename = std::format("{}{}{}.txt", BENCH_DATASET_DIR, "/Graph/", label);
      matrices[label].load_to_cpu(filename);
    }
  }
  std::cout << "\r";
  double elapsed = load_matrices_timer.measure();
  std::cout << "matrices loaded, time: " << elapsed << "s\n";

  if (load_at_gpu) {
    load_matrices_timer.mark();
    std::println("loading at VRAM");
    uint32_t initial_free_mem = get_used_memory();
    for (int i = 0; i < matrices.size(); i++) {
      uint32_t free_mem = get_used_memory();

      auto &data = matrices[i];
      if (!data._loaded) {
        continue;
      }

      data.load_to_gpu();

      if (pretransposed && data._matrix != nullptr) {
        cuBool_Index nrows, ncols;
        cuBool_Matrix_Nrows(data._matrix, &nrows);
        cuBool_Matrix_Ncols(data._matrix, &ncols);

        cuBool_Matrix_New(&data._transposed, ncols, nrows);
        cuBool_Matrix_Transpose(data._transposed, data._matrix, CUBOOL_HINT_NO);
      }

      uint32_t new_free_mem = get_used_memory();
      std::print("\r");
      std::print("matrix #{}: now used: {}, diff used: {}, actual size: {}", i, new_free_mem,
                 new_free_mem - free_mem, data.sizeMb());
      std::flush(std::cout);
    }
    elapsed = load_matrices_timer.measure();
    std::print("\r");
    std::println("matrices loaded at GPU, time: {}s, used memory: {}",
                  elapsed, get_used_memory() - initial_free_mem);
  }

  return matrices;
}

struct Query {
  std::vector<cuBool_Matrix> _graph;
  std::vector<cuBool_Matrix> _automat;

  std::vector<cuBool_Matrix> _graph_transposed;
  std::vector<cuBool_Matrix> _automat_transposed;
  bool _transposed = false;

  std::vector<cuBool_Index> _sourece_vertices;
  std::vector<cuBool_Index> _start_states;

  std::vector<cuBool_Index> _final_states;

  std::vector<uint32_t> _labels;
  std::vector<bool> _inverse_lables;
  bool _labels_inversed = false;

  bool _matrices_was_loaded = true;

  uint32_t _query_number = 0;
  Timer _query_timer;

  std::pair<bool, double> load(uint32_t query_number, const Wikidata &matrices,
                               bool preloaded = false, bool transpose = true, bool pretransposed = false);
  std::pair<uint32_t, double> execute();
  void clear();

  // load + execute + clear
  std::pair<uint32_t, double> make(uint32_t query_number, const Wikidata &matrices,
                                   bool preloaded = false, bool transpose = true) {
    if (!load(query_number, matrices, preloaded, transpose).first) {
      clear();
      return {0, 0};
    }
    auto answer = execute();
    clear();
    return answer;
  }


  ~Query() {
    clear();
  }
};

std::pair<bool, double> Query::load(uint32_t query_number, const Wikidata &matrices,
                                    bool preloaded, bool transpose, bool pretransposed) {
  _query_timer.mark();
  _query_number = query_number;

  std::string filename = std::format("{}{}{}/meta.txt", BENCH_DATASET_DIR, "/Queries/", query_number);
  std::ifstream query_file(filename);
  if (!query_file) {
    return {false, 0};
  }

  cuBool_Index source = 0, dest = 0;
  query_file >> source >> dest;
  if (source == 0 && dest == 0) {
    source = 1;
    // return {false, 0};
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
  _labels.resize(labels_number);
  _inverse_lables.resize(labels_number);
  for (int i = 0; i < labels_number; i++) {
    int label;
    query_file >> label;
    _labels[i] = std::abs(label);
    _inverse_lables[i] = label < 0;
  }

  _graph.assign(labels_number, nullptr);
  _automat.assign(labels_number, nullptr);

  _matrices_was_loaded = !preloaded;
  _transposed = transpose;

  for (int i = 0; i < labels_number; i++) {
    uint32_t label = _labels[i];
    if (!preloaded) {
      if (not matrices[label].copy_to_gpu(&_graph[i])) {
        return {false, 0};
      }
    } else {
      _graph[i] = matrices[label]._matrix;
    }

    filename = std::format("{}{}{}/{}.txt", BENCH_DATASET_DIR, "/Queries/", query_number,
                           _inverse_lables[i] ? -(int)label : (int)label);
    MatrixData data;
    if (not data.load_to_cpu(filename) || not data.copy_to_gpu(&_automat[i])) {
      return {false, 0};
    }
  }

  if (transpose) {
    _graph_transposed.reserve(_graph.size());
    if (!pretransposed) {
      for (auto label_matrix : _graph) {
        _graph_transposed.emplace_back();

        if (label_matrix == nullptr) {
          continue;
        }

        cuBool_Index nrows, ncols;
        cuBool_Matrix_Nrows(label_matrix, &nrows);
        cuBool_Matrix_Ncols(label_matrix, &ncols);

        cuBool_Matrix_New(&_graph_transposed.back(), ncols, nrows);
        cuBool_Matrix_Transpose(_graph_transposed.back(), label_matrix, CUBOOL_HINT_NO);
      }
    } else {
      for (int i = 0; i < labels_number; i++) {
        uint32_t label = _labels[i];
        _graph_transposed[i] = matrices[label]._transposed;
      }
    }

    // transpose automat matrices
    _automat_transposed.reserve(_automat.size());
    for (auto label_matrix : _automat) {
      _automat_transposed.emplace_back();
      if (label_matrix == nullptr) {
        continue;
      }

      cuBool_Index nrows, ncols;
      cuBool_Matrix_Nrows(label_matrix, &nrows);
      cuBool_Matrix_Ncols(label_matrix, &ncols);

      cuBool_Matrix_New(&_automat_transposed.back(), ncols, nrows);
      cuBool_Matrix_Transpose(_automat_transposed.back(), label_matrix, CUBOOL_HINT_NO);
    }
  }

  if (source == std::numeric_limits<cuBool_Index>::max()) {
    _start_states = std::move(inv_src_verts);
    _final_states = std::move(src_verts);
    _sourece_vertices = std::vector {dest};
    _labels_inversed = true;
  } else {
    _start_states = std::move(src_verts);
    _final_states = std::move(inv_src_verts);
    _sourece_vertices = std::vector {source};
    _labels_inversed = false;
  }

  return {true, _query_timer.measure()};
}

void Query::clear() {
  if (_matrices_was_loaded) {
    for (auto &matrix : _graph) {
      if (matrix != nullptr) {
        cuBool_Matrix_Free(matrix);
        matrix = nullptr;
      }
    }
  }

  for (auto &matrix : _automat) {
    if (matrix != nullptr) {
      cuBool_Matrix_Free(matrix);
      matrix = nullptr;
    }
  }

  // std::println("clear 3 transposed");
  if (_transposed) {
    for (auto &matrix : _graph_transposed) {
      if (matrix != nullptr) {
        cuBool_Matrix_Free(matrix);
        matrix = nullptr;
      }
    }
    for (auto &matrix : _automat_transposed) {
      if (matrix != nullptr) {
        cuBool_Matrix_Free(matrix);
        matrix = nullptr;
      }
    }
  }
}

std::pair<uint32_t, double> Query::execute() {
  std::string filename = std::format("{}/{}.txt", QUERIES_LOGS, _query_number);
  std::ofstream log_file(filename);
  std::optional<std::reference_wrapper<std::ofstream>> logger(log_file);
  logger = std::nullopt;

  cuBool_Matrix recheable = nullptr;

  static Timer make_query_timer {};

  make_query_timer.mark();
  if (_transposed) {
    recheable = regular_path_query_with_transposed(_graph, _sourece_vertices,
                                                   _automat, _start_states,
                                                   _graph_transposed,
                                                   _automat_transposed,
                                                   _inverse_lables, _labels_inversed,
                                                   logger);
  } else {
    recheable = regular_path_query(_graph, _sourece_vertices,
                                   _automat, _start_states,
                                   _inverse_lables, _labels_inversed, logger);
  }

  cuBool_Index automat_rows, graph_rows;
  cuBool_Matrix_Nrows(_graph.front(), &graph_rows);
  cuBool_Matrix_Nrows(_automat.front(), &automat_rows);

  cuBool_Vector P, F;
  cuBool_Vector_New(&P, graph_rows);
  cuBool_Vector_New(&F, automat_rows);

  cuBool_Vector_Build(F, _final_states.data(), _final_states.size(), CUBOOL_HINT_NO);
  cuBool_VxM(P, F, recheable, CUBOOL_HINT_NO);
  uint32_t answer = 0;
  cuBool_Vector_Nvals(P, &answer);

  auto time = make_query_timer.measure();

  cuBool_Vector_Free(P);
  cuBool_Vector_Free(F);
  cuBool_Matrix_Free(recheable);

  return {answer, time};
}

bool benchmark() {
  cuBool_Initialize(CUBOOL_HINT_NO);

  bool preloading = true;
  bool pretransposed_gpu = false;
  bool pretransposed = true;
  auto matrices = load_matrices(preloading, pretransposed_gpu);
  uint32_t runs_number = 10;

  std::set<uint32_t> too_big_queris = {115};
  too_big_queris = {};

  std::filesystem::create_directory(QUERIES_LOGS);
  auto total_time_file_name = "total_time_file.txt";
  std::filesystem::remove(total_time_file_name);

  for (uint32_t run = 1; run <= runs_number; run++) {
    auto result_file_name = runs_number == 1 ? std::string("result.txt")
                                             : std::format("result{}.txt", run);
    std::fstream results_file(result_file_name, std::ofstream::out);
    double total_load_time = 0;
    double total_execute_time = 0;

    std::println("run {}", run);
    std::println("query_number execute_time load_time result");
    for (uint32_t query_number = 1; query_number <= BENCH_QUERY_COUNT; query_number++) {
    // for (uint32_t query_number = 1003; query_number <= 1003; query_number++) {
      if (too_big_queris.contains(query_number)) {
        continue;
      }

      Query query;
      auto [load_successfully, load_time] =
        query.load(query_number, matrices, preloading, pretransposed, pretransposed_gpu);
      if (!load_successfully) {
        std::println("{} skipped", query_number);
        continue;
      }
      auto [result, execute_time] = query.execute();
      query.clear();

      std::println("{} {} {} {}", query_number, execute_time, load_time, result);
      std::println(results_file, "{} {} {} {}", query_number, execute_time, load_time, result);

      total_load_time += load_time;
      total_execute_time += execute_time;
    }

    std::println("\n\n");
    std::println("total load time: {}, total execute time: {}\n",
                 total_load_time, total_execute_time);

    std::ofstream total_time_file(total_time_file_name, std::ios_base::ate);
    std::println(total_time_file, "total load time: {}, total execute time: {}\n",
                                  total_load_time, total_execute_time);
  }

  cuBool_Finalize();

  return true;
}

int main(int argc, char **argv) {
  std::println("Dataset: {}\n", BENCH_DATASET_DIR);
#if 0
  std::jthread thread([](std::stop_token token) {
    auto max_mem = get_used_memory();
    std::ofstream log_file("mem_log.txt");
    while (!token.stop_requested()) {
      auto mem = get_used_memory();
      if (mem > max_mem) {
        std::println(log_file, "{}", mem);
        max_mem = mem;
      }
    }
  });
#endif

  return benchmark() ? 0 : 1;
}
