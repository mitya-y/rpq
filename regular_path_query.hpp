#include <vector>

#include <cubool.h>

void print_cubool_matrix(cuBool_Matrix matrix, std::string name = "");

cuBool_Matrix regular_path_query(
    // vector of sparse graph matrices for each label
    const std::vector<cuBool_Matrix> &graph,
    const std::vector<cuBool_Index> &source_vertices,
    // vector of sparse automat matrices for each label
    const std::vector<cuBool_Matrix> &automat,
    const std::vector<cuBool_Index> &start_states);