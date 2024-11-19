#include <stdio.h>
#include <time.h>
#include <iostream>
#include <fstream>

#include <cubool.h>

#include <fast_matrix_market/fast_matrix_market.hpp>

#include "regular_path_query.hpp"

#define LEN 512
#define MAX_LABELS 16
#define RUNS 1

#define WIKIDATA_DIR "/home/mitya/Wikidata/"
#define QUERIES_DIR "Queries/"
#define RESULTS_DIR "Results/"

#define QUERY_COUNT 660
#define PROP_COUNT 1395

bool benchmark() {
  cuBool_Initialize(CUBOOL_HINT_NO);

  struct timespec start, finish;

  // preload
  printf("Loading the matrices...\n");

  std::vector<cuBool_Matrix> G(MAX_LABELS);
  std::vector<cuBool_Matrix> GS(PROP_COUNT + 1);
  std::vector<cuBool_Matrix> R(MAX_LABELS);

  for (int64_t label = 1; label <= PROP_COUNT; label++) {
    std::string filename = WIKIDATA_DIR + std::to_string(label) + ".txt";
    std::ifstream file(filename);
    if (not file) {
      continue;
    }

    int64_t nrows = 0, ncols = 0;
    std::vector<cuBool_Index> rows, cols;
    std::vector<bool> vals;
    fast_matrix_market::read_matrix_market_triplet(file, nrows, ncols, rows, cols, vals);

    cuBool_Matrix *matrix = &GS[label];

    cuBool_Matrix_New(matrix, nrows, ncols);
    cuBool_Matrix_Build(*matrix, rows.data(), cols.data(), vals.size(), CUBOOL_HINT_NO);
  }

  // TODO: calculate memory usage

  printf("Loading done!\n");

  // Run one more time to fill in caches.
  for (int runs_number = 0; runs_number <= RUNS; runs_number++) {
    printf("run number: %d\n", runs_number);

    for (int query = 0; query < QUERY_COUNT + 1; query++) {
      std::string filename = QUERIES_DIR + std::to_string(query) + ".txt";
      std::ifstream file(filename);
      if (not file) {
        continue;
      }

      cuBool_Index source = 0, dest = 0;
      file >> source >> dest;

      if (source == 0 && dest == 0) {
        printf("%d, skipped\n", query);
        continue;
      }

      int64_t nqS = 0;
      std::vector<cuBool_Index> qS {};
      file >> nqS;
      qS.resize(nqS);
      for (int i = 0; i < nqS; i++) {
        file >> qS[i];
        qS[i]--;
      }

      uint64_t nqF = 0;
      std::vector<cuBool_Index> qF {};
      file >> nqF;
      qF.resize(nqF);
      for (int i = 0; i < nqF; i++) {
        file >> qF[i];
        qF[i]--;
      }

      source--;
      dest--;

      int64_t label;
      int64_t labels[MAX_LABELS];
      bool inverse_labels[MAX_LABELS];
      int64_t nl;

      file >> nl;
      nl = 0;

      while (file) {
        file >> label;
        labels[nl] = label;
        inverse_labels[nl++] = label < 0;
      }

      for (int i = 0; i < MAX_LABELS; i++) {
        G[i] = NULL;
        R[i] = NULL;
      }

      for (int i = 0; i < nl; i++) {
        uint64_t label = labels[i] > 0 ? labels[i] : -labels[i];
        if (GS[label]) {
          G[i] = GS[label];
          continue;
        }

        std::string filename = WIKIDATA_DIR + std::to_string(label) + ".txt";
        std::ifstream file(filename);
        if (not file) {
          continue;
        }

        int64_t nrows = 0, ncols = 0;
        std::vector<cuBool_Index> rows, cols;
        std::vector<bool> vals;
        fast_matrix_market::read_matrix_market_triplet(file, nrows, ncols, rows, cols, vals);

        cuBool_Matrix *matrix = &GS[label];

        cuBool_Matrix_New(matrix, nrows, ncols);
        cuBool_Matrix_Build(*matrix, rows.data(), cols.data(), vals.size(), CUBOOL_HINT_NO);
        G[i] = GS[label];
      }

      for (int i = 0; i < nl; i++) {
        std::string filename = QUERIES_DIR + std::to_string(query) + '/' + std::to_string(labels[i]) + ".txt";
        std::ifstream file(filename);
        if (not file) {
          continue;
        }

        int64_t nrows = 0, ncols = 0;
        std::vector<cuBool_Index> rows, cols;
        std::vector<bool> vals;
        fast_matrix_market::read_matrix_market_triplet(file, nrows, ncols, rows, cols, vals);

        cuBool_Matrix *matrix = &R[i];

        cuBool_Matrix_New(matrix, nrows, ncols);
        cuBool_Matrix_Build(*matrix, rows.data(), cols.data(), vals.size(), CUBOOL_HINT_NO);
      }

      std::vector<cuBool_Index> S { source };
      std::vector<cuBool_Index> D { dest };

      clock_gettime(CLOCK_MONOTONIC, &start);

      G.resize(nl);
      R.resize(nl);

      cuBool_Matrix reachable = source != -1
        ? regular_path_query(G, qS, R, S)
        : regular_path_query(G, qS, R, D);

      cuBool_Index ng;
      cuBool_Index nr;

      cuBool_Matrix_Nrows(G[0], &ng);
      cuBool_Matrix_Nrows(R[0], &nr);

      cuBool_Vector F, P;
      cuBool_Vector_New(&F, nr);
      cuBool_Vector_New(&P, ng);

      if (source != -1) {
        for (int i = 0; i < nqF; i++) {
          cuBool_Vector_SetElement(F, qF[i]);
        }
      } else {
        for (int i = 0; i < nqS; i++) {
          cuBool_Vector_SetElement(F, qS[i]);
        }
      }

      cuBool_Index entries = 0;
      cuBool_VxM(P, F, reachable, CUBOOL_HINT_NO);
      cuBool_Vector_Nvals(P, &entries);

      // Export all the nodes matching final FA states.
      clock_gettime(CLOCK_MONOTONIC, &finish);
      double elapsed = (finish.tv_sec - start.tv_sec) * 1000000.0 + (finish.tv_nsec - start.tv_nsec) / 1000.0;

      printf("%d,%.0lf,%d\n", query, elapsed, entries);

      if (runs_number != 0) {
        std::string filename = RESULTS_DIR + std::to_string(query) + ".txt";
        std::ofstream file(filename);
        if (not file) {
          continue;
        }
        file << elapsed << ' ' << entries << '\n';
      }
      cuBool_Matrix_Free(reachable);
    }
  }

  return true;
}

