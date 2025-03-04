#include <future>
#include <cassert>
#include <iostream>
#include <print>
#include <set>
#include <numeric>
#include <ranges>

#include "cubool.h"
#include "regular_path_query.hpp"
#include "timer.hpp"

#include "BS_thread_pool.hpp"

static Timer rpq_timer {};

cuBool_Matrix par_regular_path_query(
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

  BS::thread_pool pool;

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

    futures.push_back(pool.submit_task([matrix = graph_transpsed.back(), label_matrix]() {
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

      futures.push_back(pool.submit_task([matrix = automat_transpsed.back(), label_matrix]() {
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

  for (auto &future : futures) {
    status = future.get();
    assert(status == CUBOOL_STATUS_SUCCESS);
  }

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
          cuBool_Matrix automat_matrix = all_labels_are_inversed ? automat[i] : automat_transpsed[i];
          status = cuBool_MxM(util, automat_matrix, frontier, CUBOOL_HINT_NO);
          if (status != CUBOOL_STATUS_SUCCESS) {
            return status;
          }

          // we want: next_frontier += (symbol_frontier * graph[i]) & (!reachible)
          cuBool_Matrix graph_matrix = inversed_labels[i] ? graph_transpsed[i] : graph[i];
          status = cuBool_MxM(result, util, graph_matrix, CUBOOL_HINT_NO);
          if (status != CUBOOL_STATUS_SUCCESS) {
            return status;
          }

#if 0
          // apply invert mask
          status = cuBool_Matrix_EWiseMulInverted(util, result, reacheble, CUBOOL_HINT_NO);
          if (status != CUBOOL_STATUS_SUCCESS) {
            return status;
          }
          std::swap(result, util);
#endif

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

    // TODO: this is bag, fix it
    // UPD: maybe not bug, maybe it is feature
    // apply invert mask
    status = cuBool_Matrix_EWiseMulInverted(util, next_frontier, reacheble, CUBOOL_HINT_NO);
    std::swap(util, next_frontier);

    // this must be accumulate with mask and save old value: reacheble += next_frontier | reacheble
    status = cuBool_Matrix_EWiseAdd(util, reacheble, next_frontier, CUBOOL_HINT_NO);
    assert(status == CUBOOL_STATUS_SUCCESS);
    std::swap(util, reacheble);

    cuBool_Matrix_Nvals(next_frontier, &states);

    // print_cubool_matrix(next_frontier, "next_frontier", true);
    // print_cubool_matrix(reacheble, "reacheble", true);
  }

  std::println(out, "load time = {}, execute_time = {}", load_time, rpq_timer.measure());

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
