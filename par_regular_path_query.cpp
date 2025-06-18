#include <future>
#include <cassert>
#include <iostream>
#include <print>
#include <set>
#include <numeric>
#include <ranges>

#include "regular_path_query.hpp"
#include "timer.hpp"

#include "BS_thread_pool.hpp"

cuBool_Matrix par_regular_path_query_with_transposed(
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

  Timer rpq_timer {};
  rpq_timer.mark();

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

  auto load_time = rpq_timer.measure();

  const auto label_number = std::min(graph.size(), automat.size());

  std::vector<cuBool_Matrix> result_label_matrices;
  result_label_matrices.resize(label_number);
  for (auto &matrix : result_label_matrices) {
    status = cuBool_Matrix_New(&matrix, automat_nodes_number, graph_nodes_number);
    assert(status == CUBOOL_STATUS_SUCCESS);
  }

  std::vector<cuBool_Matrix> util_label_matrices;
  util_label_matrices.resize(label_number);
  for (auto &matrix : util_label_matrices) {
    status = cuBool_Matrix_New(&matrix, automat_nodes_number, graph_nodes_number);
    assert(status == CUBOOL_STATUS_SUCCESS);
  }

  BS::thread_pool pool;
  std::vector<std::future<cuBool_Status>> futures;
  futures.reserve(label_number);

  while (states > 0) {
    std::swap(frontier, next_frontier);

    futures.clear();
    for (int i = 0; i < label_number; i++) {
      if (graph[i] == nullptr || automat[i] == nullptr) {
        continue;
      }

      futures.push_back(pool.submit_task(
        [&, result = result_label_matrices[i], util = util_label_matrices[i], i]() mutable {
          cuBool_Matrix automat_matrix = all_labels_are_inversed ? automat[i] : automat_transposed[i];
          status = cuBool_MxM(util, automat_matrix, frontier, CUBOOL_HINT_NO);
          if (status != CUBOOL_STATUS_SUCCESS) {
            return status;
          }

          // we want: next_frontier += (symbol_frontier * graph[i]) & (!reachible)
          cuBool_Matrix graph_matrix = inversed_labels[i] ? graph_transposed[i] : graph[i];
          status = cuBool_MxM(result, util, graph_matrix, CUBOOL_HINT_NO);
          if (status != CUBOOL_STATUS_SUCCESS) {
            return status;
          }

          return status;
      }));
    }

    for (auto &future : futures) {
      status = future.get();
      assert(status == CUBOOL_STATUS_SUCCESS);
    }

    auto size = result_label_matrices.size();
    while (size > 1) {
      auto pairs_number = size / 2;
      for (std::size_t i = 0; i < pairs_number; i++) {
        pool.detach_task(
        [&a = result_label_matrices[i],
          b = result_label_matrices[size - 1 - i],
         &c = util_label_matrices[i]]() {
          cuBool_Matrix_EWiseAdd(c, a, b, CUBOOL_HINT_NO);
          std::swap(a, c);
        });
      }
      pool.wait();
      size = pairs_number + (size % 2);
    }
    std::swap(next_frontier, result_label_matrices[0]);

    assert(util_label_matrices.size() > 0);
    auto &util = util_label_matrices[0];

    status = cuBool_Matrix_EWiseMulInverted(util, next_frontier, reacheble, CUBOOL_HINT_NO);
    std::swap(util, next_frontier);

    status = cuBool_Matrix_EWiseAdd(util, reacheble, next_frontier, CUBOOL_HINT_NO);
    assert(status == CUBOOL_STATUS_SUCCESS);
    std::swap(util, reacheble);

    cuBool_Matrix_Nvals(next_frontier, &states);
  }

  if (out.has_value()) {
    auto &out_value = out.value().get();
    std::println(out_value, "load time = {}, execute_time = {}", load_time, rpq_timer.measure());
  }

  // free matrix necessary for algorithm
  for (auto &matrix : result_label_matrices) {
    status = cuBool_Matrix_Free(matrix);
    assert(status == CUBOOL_STATUS_SUCCESS);
  }

  for (auto &matrix : util_label_matrices) {
    status = cuBool_Matrix_Free(matrix);
    assert(status == CUBOOL_STATUS_SUCCESS);
  }

  cuBool_Matrix_Free(next_frontier);
  cuBool_Matrix_Free(frontier);
  cuBool_Matrix_Free(symbol_frontier);

  return reacheble;
}

cuBool_Matrix par_regular_path_query(
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

  auto result = par_regular_path_query_with_transposed(
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