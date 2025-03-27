#include <vector>

#include <cubool.h>

void print_cubool_matrix(cuBool_Matrix matrix, std::string name = "", bool print_full = false);
void print_cubool_vector(cuBool_Vector vector, std::string name = "");

cuBool_Matrix regular_path_query_with_transposed(
  // vector of sparse graph matrices for each label
  const std::vector<cuBool_Matrix> &graph, const std::vector<cuBool_Index> &source_vertices,
  // vector of sparse automat matrices for each label
  const std::vector<cuBool_Matrix> &automat, const std::vector<cuBool_Index> &start_states,
  // transposed matrices for graph and automat
  const std::vector<cuBool_Matrix> &graph_transposed,
  const std::vector<cuBool_Matrix> &automat_transposed,

  const std::vector<bool> &inversed_labels = {}, bool all_labels_are_inversed = false,
  std::optional<std::reference_wrapper<std::ostream>> out = std::nullopt);

cuBool_Matrix regular_path_query(
  // vector of sparse graph matrices for each label
  const std::vector<cuBool_Matrix> &graph, const std::vector<cuBool_Index> &source_vertices,
  // vector of sparse automat matrices for each label
  const std::vector<cuBool_Matrix> &automat, const std::vector<cuBool_Index> &start_states,

  const std::vector<bool> &inversed_labels = {}, bool all_labels_are_inversed = false,
  std::optional<std::reference_wrapper<std::ostream>> out = std::nullopt);

// concurrent version
cuBool_Matrix par_regular_path_query_with_transposed(
  // vector of sparse graph matrices for each label
  const std::vector<cuBool_Matrix> &graph, const std::vector<cuBool_Index> &source_vertices,
  // vector of sparse automat matrices for each label
  const std::vector<cuBool_Matrix> &automat, const std::vector<cuBool_Index> &start_states,
  // transposed matrices for graph and automat
  const std::vector<cuBool_Matrix> &graph_transposed,
  const std::vector<cuBool_Matrix> &automat_transposed,

  const std::vector<bool> &inversed_labels = {}, bool all_labels_are_inversed = false,
  std::optional<std::reference_wrapper<std::ostream>> out = std::nullopt);

// concurrent version
cuBool_Matrix par_regular_path_query(
  // vector of sparse graph matrices for each label
  const std::vector<cuBool_Matrix> &graph, const std::vector<cuBool_Index> &source_vertices,
  // vector of sparse automat matrices for each label
  const std::vector<cuBool_Matrix> &automat, const std::vector<cuBool_Index> &start_states,

  const std::vector<bool> &inversed_labels = {}, bool all_labels_are_inversed = false,
  std::optional<std::reference_wrapper<std::ostream>> out = std::nullopt);


bool benchmark();
