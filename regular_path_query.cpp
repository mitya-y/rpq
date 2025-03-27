#include <cassert>
#include <cstdint>
#include <iostream>
#include <print>
#include <set>

#include "cubool.h"
#include "regular_path_query.hpp"
#include "timer.hpp"

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

cuBool_Matrix regular_path_query_with_transposed(
  // vector of sparse graph matrices for each label
  const std::vector<cuBool_Matrix> &graph, const std::vector<cuBool_Index> &source_vertices,
  // vector of sparse automat matrices for each label
  const std::vector<cuBool_Matrix> &automat, const std::vector<cuBool_Index> &start_states,
  // transposed matrices for graph and automat
  const std::vector<cuBool_Matrix> &graph_transposed,
  const std::vector<cuBool_Matrix> &automat_transposed,

  const std::vector<bool> &inversed_labels_input, bool all_labels_are_inversed,
  std::optional<std::reference_wrapper<std::ostream>> out) {
  cuBool_Status status;

  auto inversed_labels = inversed_labels_input;
  inversed_labels.resize(std::max(graph.size(), automat.size()));

  for (uint32_t i = 0; i < inversed_labels.size(); i++) {
    bool is_inverse = inversed_labels[i];
    is_inverse ^= all_labels_are_inversed;
    inversed_labels[i] = is_inverse;
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

  Timer add_timer, mxm_timer;
  double add_time = 0, mxm_time = 0;

  uint32_t iter_number = 0;
  const auto label_number = std::min(graph.size(), automat.size());
  while (states > 0) {
    iter_number++;

    std::swap(frontier, next_frontier);

    // clear next_frontier
    status = cuBool_Matrix_Build(next_frontier, nullptr, nullptr, 0, CUBOOL_HINT_NO);
    assert(status == CUBOOL_STATUS_SUCCESS);

    for (int i = 0; i < label_number; i++) {
      if (graph[i] == nullptr || automat[i] == nullptr) {
        continue;
      }

      mxm_timer.mark();
      cuBool_Matrix automat_matrix = all_labels_are_inversed ? automat[i] : automat_transposed[i];
      status = cuBool_MxM(symbol_frontier, automat_matrix, frontier, CUBOOL_HINT_NO);
      assert(status == CUBOOL_STATUS_SUCCESS);

      // TODO: check states here

      // we want: next_frontier += (symbol_frontier * graph[i]) & (!reachible)
      // mult 2 matrices
      cuBool_Matrix graph_matrix = inversed_labels[i] ? graph_transposed[i] : graph[i];
      status = cuBool_MxM(next_frontier, symbol_frontier, graph_matrix, CUBOOL_HINT_ACCUMULATE);
      assert(status == CUBOOL_STATUS_SUCCESS);
      mxm_time += mxm_timer.measure();
      // apply invert mask
      // TODO: maybe apply mask after
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

  if (out.has_value()) {
    auto &out_value = out.value().get();

    std::println(out_value, "iter_number = {}", iter_number);
    for (auto label_matrix : graph) {
      cuBool_Index nvals, nrows, ncols;
      cuBool_Matrix_Nvals(label_matrix, &nvals);
      cuBool_Matrix_Nrows(label_matrix, &nrows);
      cuBool_Matrix_Ncols(label_matrix, &ncols);
      std::println(out_value, "{} {} {}", nvals, nrows, ncols);
    }
  }

  // std::println("add time = {}", add_time);
  // std::println("mxm time = {}", mxm_time);

  // free matrix necessary for algorithm
  cuBool_Matrix_Free(next_frontier);
  cuBool_Matrix_Free(frontier);
  cuBool_Matrix_Free(symbol_frontier);
  cuBool_Matrix_Free(result);

  return reacheble;
}


cuBool_Matrix regular_path_query(
  // vector of sparse graph matrices for each label
  const std::vector<cuBool_Matrix> &graph, const std::vector<cuBool_Index> &source_vertices,
  // vector of sparse automat matrices for each label
  const std::vector<cuBool_Matrix> &automat, const std::vector<cuBool_Index> &start_states,
  // work with inverted labels
  const std::vector<bool> &inversed_labels_input, bool all_labels_are_inversed,
  // for debug
  std::optional<std::reference_wrapper<std::ostream>> out) {
  cuBool_Status status;

  // transpose graph matrices
  std::vector<cuBool_Matrix> graph_transposed;
  graph_transposed.reserve(graph.size());
  for (uint32_t i = 0; i < graph.size(); i++) {
    graph_transposed.emplace_back();

    auto label_matrix = graph[i];
    if (label_matrix == nullptr) {
      continue;
    }

    cuBool_Index nrows, ncols;
    cuBool_Matrix_Nrows(label_matrix, &nrows);
    cuBool_Matrix_Ncols(label_matrix, &ncols);

    status = cuBool_Matrix_New(&graph_transposed.back(), ncols, nrows);
    assert(status == CUBOOL_STATUS_SUCCESS);
    status = cuBool_Matrix_Transpose(graph_transposed.back(), label_matrix, CUBOOL_HINT_NO);
    assert(status == CUBOOL_STATUS_SUCCESS);
  }

  // transpose automat matrices
  std::vector<cuBool_Matrix> automat_transposed;
  automat_transposed.reserve(automat.size());
  for (auto label_matrix : automat) {
    automat_transposed.emplace_back();
    if (label_matrix == nullptr) {
      continue;
    }

    cuBool_Index nrows, ncols;
    cuBool_Matrix_Nrows(label_matrix, &nrows);
    cuBool_Matrix_Ncols(label_matrix, &ncols);

    status = cuBool_Matrix_New(&automat_transposed.back(), ncols, nrows);
    assert(status == CUBOOL_STATUS_SUCCESS);
    status = cuBool_Matrix_Transpose(automat_transposed.back(), label_matrix, CUBOOL_HINT_NO);
    assert(status == CUBOOL_STATUS_SUCCESS);
  }

  auto result = regular_path_query_with_transposed(
    graph, source_vertices,
    automat, start_states,
    graph_transposed, automat_transposed,
    inversed_labels_input, all_labels_are_inversed,
    out);

  for (cuBool_Matrix matrix : graph_transposed) {
    if (matrix != nullptr) {
      cuBool_Matrix_Free(matrix);
    }
  }
  for (cuBool_Matrix matrix : automat_transposed) {
    if (matrix != nullptr) {
      cuBool_Matrix_Free(matrix);
    }
  }

  return result;
}
