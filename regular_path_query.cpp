#include <future>
#include <cassert>
#include <iostream>
#include <print>
#include <set>

#include "cubool.h"
#include "regular_path_query.hpp"
#include "timer.hpp"

static Timer rpq_timer {};

void print_cubool_matrix(cuBool_Matrix matrix, std::string name, bool print_full) {
  if (name != "") {
    std::cout << name << std::endl;
  }

  cuBool_Index nvals;
  cuBool_Matrix_Nvals(matrix, &nvals);
  std::vector<cuBool_Index> rows(nvals), cols(nvals);
  cuBool_Matrix_ExtractPairs(matrix, rows.data(), cols.data(), &nvals);

  if (!print_full) {
    for (int i = 0; i < nvals; i++) {
      printf("(%d, %d)\n", rows[i], cols[i]);
    }
    return;
  }

  std::set<std::pair<cuBool_Index, cuBool_Index>> indexes {};
  for (int i = 0; i < nvals; i++) {
    indexes.insert({rows[i], cols[i]});
  }

  cuBool_Index nrows, ncols;
  cuBool_Matrix_Ncols(matrix, &ncols);
  cuBool_Matrix_Nrows(matrix, &nrows);

  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      std::cout << (indexes.contains({i, j}) ? '1' : '0') << ' ';
    }
    std::cout << "\n";
  }
}

void print_cubool_vector(cuBool_Vector vector, std::string name) {
  if (name != "") {
    std::cout << name << ": ";
  }

  cuBool_Index nvals;
  cuBool_Vector_Nvals(vector, &nvals);
  std::vector<cuBool_Index> row(nvals);
  cuBool_Vector_ExtractValues(vector, row.data(), &nvals);

  printf("{");
  for (int i = 0; i < nvals; i++) {
    printf("%d", row[i]);
    if (i != nvals - 1) {
      printf(", ");
    }
  }
  printf("}, size = %d\n", nvals);
}


cuBool_Matrix regular_path_query(
  // vector of sparse graph matrices for each label
  const std::vector<cuBool_Matrix> &graph, const std::vector<cuBool_Index> &source_vertices,
  // vector of sparse automat matrices for each label
  const std::vector<cuBool_Matrix> &automat, const std::vector<cuBool_Index> &start_states,
  // work with inverted labels
  const std::vector<bool> &inversed_labels_input, bool all_labels_are_inversed,
  // for debug
  std::ostream &out) {
  cuBool_Status status;

  rpq_timer.mark();

  auto inversed_labels = inversed_labels_input;
  inversed_labels.resize(std::max(graph.size(), automat.size()));

  for (uint32_t i = 0; i < inversed_labels.size(); i++) {
    bool is_inverse = inversed_labels[i];
    is_inverse ^= all_labels_are_inversed;
    inversed_labels[i] = is_inverse;
  }

  constexpr auto transpose_launch_policy = std::launch::deferred;
  std::vector<std::future<cuBool_Status>> futures;

  // transpose graph matrices
  std::vector<cuBool_Matrix> graph_transpsed;
  graph_transpsed.reserve(graph.size());
  for (uint32_t i = 0; i < graph.size(); i++) {
    graph_transpsed.emplace_back();

    // we use transposed graph matrix only if label is inversed
    if (!inversed_labels[i]) {
      continue;
    }

    auto label_matrix = graph[i];
    if (label_matrix == nullptr) {
      continue;
    }

    if (!inversed_labels[i]) {
      continue;
    }

    cuBool_Index nrows, ncols;
    cuBool_Matrix_Nrows(label_matrix, &nrows);
    cuBool_Matrix_Ncols(label_matrix, &ncols);

    status = cuBool_Matrix_New(&graph_transpsed.back(), ncols, nrows);
    assert(status == CUBOOL_STATUS_SUCCESS);

    futures.push_back(std::async(transpose_launch_policy,
      [matrix = graph_transpsed.back(), label_matrix]() {
        auto status = cuBool_Matrix_Transpose(matrix, label_matrix, CUBOOL_HINT_NO);
        return status;
      }));
  }

  // transpose automat matrices
  std::vector<cuBool_Matrix> automat_transpsed;
  automat_transpsed.reserve(automat.size());
  // if all lables are transposed we don't use transposed automat matrices
  if (!all_labels_are_inversed) {
    for (auto label_matrix : automat) {
      automat_transpsed.emplace_back();
      if (label_matrix == nullptr) {
        continue;
      }

      cuBool_Index nrows, ncols;
      cuBool_Matrix_Nrows(label_matrix, &nrows);
      cuBool_Matrix_Ncols(label_matrix, &ncols);

      status = cuBool_Matrix_New(&automat_transpsed.back(), ncols, nrows);
      assert(status == CUBOOL_STATUS_SUCCESS);

      futures.push_back(std::async(transpose_launch_policy,
        [matrix = automat_transpsed.back(), label_matrix]() {
          auto status = cuBool_Matrix_Transpose(matrix, label_matrix, CUBOOL_HINT_NO);
          return status;
        }));
    }
  }

  cuBool_Index graph_nodes_number = 0;
  cuBool_Index automat_nodes_number = 0;

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
  status = cuBool_Matrix_New(&reacheble, automat_nodes_number, graph_nodes_number);
  assert(status == CUBOOL_STATUS_SUCCESS);

  // allocate neccessary for algorithm matrices
  cuBool_Matrix frontier {}, symbol_frontier {}, next_frontier {};
  status = cuBool_Matrix_New(&next_frontier, automat_nodes_number, graph_nodes_number);
  assert(status == CUBOOL_STATUS_SUCCESS);
  status = cuBool_Matrix_New(&frontier, automat_nodes_number, graph_nodes_number);
  assert(status == CUBOOL_STATUS_SUCCESS);
  status = cuBool_Matrix_New(&symbol_frontier, automat_nodes_number, graph_nodes_number);
  assert(status == CUBOOL_STATUS_SUCCESS);

  // init start values of algorithm matricies
  for (const auto state : start_states) {
    for (const auto vert : source_vertices) {
      assert(state < automat_nodes_number);
      assert(vert < graph_nodes_number);
      cuBool_Matrix_SetElement(next_frontier, state, vert);
      cuBool_Matrix_SetElement(reacheble, state, vert);
    }
  }

  cuBool_Index states = source_vertices.size();

  // temporary matrix for write result of cubool functions
  cuBool_Matrix result;
  status = cuBool_Matrix_New(&result, automat_nodes_number, graph_nodes_number);
  assert(status == CUBOOL_STATUS_SUCCESS);

  auto load_time = rpq_timer.measure();

  Timer add_timer, mxm_timer;
  double add_time = 0, mxm_time = 0;

  for (auto &future : futures) {
    assert(future.get() == CUBOOL_STATUS_SUCCESS);
  }

  const auto label_number = std::min(graph.size(), automat.size());
  while (states > 0) {
    std::swap(frontier, next_frontier);

    // clear next_frontier
    status = cuBool_Matrix_Build(next_frontier, nullptr, nullptr, 0, CUBOOL_HINT_NO);
    assert(status == CUBOOL_STATUS_SUCCESS);

    for (int i = 0; i < label_number; i++) {
      if (graph[i] == nullptr || automat[i] == nullptr) {
        continue;
      }

      mxm_timer.mark();
      cuBool_Matrix automat_matrix = all_labels_are_inversed ? automat[i] : automat_transpsed[i];
      status = cuBool_MxM(symbol_frontier, automat_matrix, frontier, CUBOOL_HINT_NO);
      assert(status == CUBOOL_STATUS_SUCCESS);

      // TODO: check states here

      // we want: next_frontier += (symbol_frontier * graph[i]) & (!reachible)
      // mult 2 matrices
      cuBool_Matrix graph_matrix = inversed_labels[i] ? graph_transpsed[i] : graph[i];
      status = cuBool_MxM(next_frontier, symbol_frontier, graph_matrix, CUBOOL_HINT_ACCUMULATE);
      assert(status == CUBOOL_STATUS_SUCCESS);
      mxm_time += mxm_timer.measure();
      // apply invert mask
      status = cuBool_Matrix_EWiseMulInverted(result, next_frontier, reacheble, CUBOOL_HINT_NO);
      assert(status == CUBOOL_STATUS_SUCCESS);
      std::swap(result, next_frontier);
    }

    // this must be accumulate with mask and save old value: reacheble += next_frontier & reacheble
    add_timer.mark();
    status = cuBool_Matrix_EWiseAdd(result, reacheble, next_frontier, CUBOOL_HINT_NO);
    add_time += add_timer.measure();
    assert(status == CUBOOL_STATUS_SUCCESS);
    std::swap(result, reacheble);

    cuBool_Matrix_Nvals(next_frontier, &states);
  }

  std::println(out, "load time = {}, execute_time = {}", load_time, rpq_timer.measure());

  std::println("add time = {}", add_time);
  std::println("mxm time = {}", mxm_time);

  // free matrix necessary for algorithm
  cuBool_Matrix_Free(next_frontier);
  cuBool_Matrix_Free(frontier);
  cuBool_Matrix_Free(symbol_frontier);
  cuBool_Matrix_Free(result);

  for (cuBool_Matrix matrix : graph_transpsed) {
    if (matrix != nullptr) {
      cuBool_Matrix_Free(matrix);
    }
  }
  for (cuBool_Matrix matrix : automat_transpsed) {
    if (matrix != nullptr) {
      cuBool_Matrix_Free(matrix);
    }
  }

  return reacheble;
}
