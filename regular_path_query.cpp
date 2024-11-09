#include <iostream>
#include <cassert>

#include "regular_path_query.hpp"

void print_cubool_matrix(cuBool_Matrix matrix, std::string name) {
  if (name != "") {
    std::cout << name << std::endl;
  }

  cuBool_Index nvals;
  cuBool_Matrix_Nvals(matrix, &nvals);
  std::vector<cuBool_Index> rows(nvals), cols(nvals);
  cuBool_Matrix_ExtractPairs(matrix, rows.data(), cols.data(), &nvals);

  for (int i = 0; i < nvals; i++) {
    printf("(%d, %d)\n", rows[i], cols[i]);
  }
}

cuBool_Matrix regular_path_query(
    // vector of sparse graph matrices for each label
    const std::vector<cuBool_Matrix> &graph,
    const std::vector<cuBool_Index> &source_vertices,
    // vector of sparse automat matrices for each label
    const std::vector<cuBool_Matrix> &automat,
    const std::vector<cuBool_Index> &start_states) {
  cuBool_Status status;

  // this is pointers to normal matrix.
  cuBool_Matrix frontier {}, symbol_frontier {}, next_frontier {};

  cuBool_Index graph_nodes_number = 0;
  cuBool_Index automat_nodes_number = 0;

  // transpose graph matrices
  std::vector<cuBool_Matrix> graph_transpsed;
  graph_transpsed.reserve(graph.size());
  for (auto label_matrix : graph) {
    // TODO call new
    graph_transpsed.emplace_back();
    cuBool_Matrix_Transpose(graph_transpsed.back(), label_matrix, CUBOOL_HINT_NO);
  }

  // transpose automat matrices
  std::vector<cuBool_Matrix> automat_transpsed;
  automat_transpsed.reserve(graph.size());
  for (auto label_matrix : automat) {
    cuBool_Index nrows, ncols;
    cuBool_Matrix_Nrows(label_matrix, &nrows);
    cuBool_Matrix_Ncols(label_matrix, &ncols);

    automat_transpsed.emplace_back();
    cuBool_Matrix_New(&automat_transpsed.back(), nrows, ncols);
    cuBool_Matrix_Transpose(automat_transpsed.back(), label_matrix, CUBOOL_HINT_NO);
  }

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
  cuBool_Matrix_New(&reacheble, automat_nodes_number, graph_nodes_number);

  // allocate neccessary for algorithm matrices
  cuBool_Matrix_New(&next_frontier, automat_nodes_number, graph_nodes_number);
  cuBool_Matrix_New(&frontier, automat_nodes_number, graph_nodes_number);
  cuBool_Matrix_New(&symbol_frontier, automat_nodes_number, graph_nodes_number);

  // init start values of algorithm matricies
  for (const auto state : start_states) {
    for (const auto vert : source_vertices) {
      assert(vert < graph_nodes_number);
      cuBool_Matrix_SetElement(next_frontier, state, vert);
      cuBool_Matrix_SetElement(reacheble, state, vert);
    }
  }

  cuBool_Index states = source_vertices.size();

  cuBool_Matrix result;
  cuBool_Matrix_New(&result, automat_nodes_number, graph_nodes_number);

  const auto label_number = std::min(graph.size(), automat.size());
  while (states > 0) {
    std::swap(frontier, next_frontier);

    // uint32_t nvals;
    // cuBool_Matrix_Nvals(reacheble, &nvals);
    // std::cout << "nvals = " << nvals << ", states = " << states << std::endl;

    // clear next_frontier
    cuBool_Matrix_Build(next_frontier, nullptr, nullptr, 0, CUBOOL_HINT_NO);

    for (int i = 0; i < label_number; i++) {
      if (graph[i] == nullptr || automat[i] == nullptr) {
        continue;
      }

      // is this overwrite symbol_frontier? not :(
      status = cuBool_MxM(symbol_frontier, automat_transpsed[i], frontier, CUBOOL_HINT_NO);
      assert(status == CUBOOL_STATUS_SUCCESS);

      // this must be mult with mask: next_frontier += (symbol_frontier * graph[i]) & (!reachible)
      status = cuBool_MxM(next_frontier, symbol_frontier, graph[i], CUBOOL_HINT_ACCUMULATE);
      assert(status == CUBOOL_STATUS_SUCCESS);

      // apply mask
      cuBool_Matrix_ApplyInverted(result, next_frontier, reacheble, CUBOOL_HINT_NO);
      std::swap(result, next_frontier);

      // cuBool_Matrix_Nvals(symbol_frontier, &nvals);
      // std::cout << "nvals symbol_frontier = " << nvals << std::endl;
      // cuBool_Matrix_Nvals(next_frontier, &nvals);
      // std::cout << "nvals next_frontier = " << nvals << std::endl;
    }

    // this must be accumulate with mask and save old value: reacheble += next_frontier & reacheble
    //
    cuBool_Matrix_EWiseAdd(result, reacheble, next_frontier, CUBOOL_HINT_NO);
    std::swap(result, reacheble);

    cuBool_Matrix_Nvals(next_frontier, &states);
  }

  cuBool_Matrix_Free(next_frontier);
  cuBool_Matrix_Free(frontier);
  cuBool_Matrix_Free(symbol_frontier);
  cuBool_Matrix_Free(result);

  return reacheble;
}
