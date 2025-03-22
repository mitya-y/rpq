import matplotlib.pyplot as plt
import math

class Matrix:
    nvals: int
    nrows: int
    ncols: int

    def __init__(self, nvals, nrows, ncols):
        self.nvals = nvals
        self.nrows = nrows
        self.ncols = ncols

class Query:
    # matrices: list[Matrix]
    number: int
    execute_time: float
    load_time: float
    iter_number: int
    result: int

    nvals_per_iter: int

    def __init__(self):
        self.matrices: list[Matrix] = []

def load_queries() -> list[Query]:
    queries: list[Query] = []

    with open("data/result.txt") as file:
        for line in file:
            queries.append(Query())
            query = queries[-1]

            values = line.split()
            query.number = int(values[0])
            query.execute_time = float(values[1])
            query.load_time = float(values[2])
            query.result = int(values[3])

            query_file = open(f"data/queries_logs/{query.number}.txt")
            query.iter_number = int(query_file.readline().split('=')[-1])
            for matrix_line in query_file:
                query.matrices.append(Matrix(*map(int, matrix_line.split())))

            query.nvals_per_iter = sum(matrix.nvals for matrix in query.matrices)

    return queries

def nvals_infl(queries: list[Query], min_bound = -math.inf):
    values = [(query.nvals_per_iter, query.execute_time)
              for query in queries if query.execute_time > min_bound]
    values.sort(key=lambda val: val[0])
    x = [val[0] for val in values]
    y = [val[1] for val in values]
    plt.figure(figsize=(8, 4))
    # 'o-' for points + lines, none for only lines, ro for points
    plt.plot(x, y, 'ro', label="exec_time(nvals)", markersize=4)
    plt.legend()
    plt.show()

def iter_number_infl(queries: list[Query], min_bound = -math.inf):
    values = [(query.iter_number, query.execute_time)
              for query in queries if query.execute_time > min_bound]
    values.sort(key=lambda val: val[0])
    x = [val[0] for val in values]
    y = [val[1] for val in values]
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, 'ro', label="exec_time(iter_number)", markersize=4)
    plt.legend()
    plt.show()

def unique_ncols(queries: list[Query]):
    unique_ncols = set()
    for query in queries:
        for matrix in query.matrices:
            unique_ncols.add(matrix.nrows)
            unique_ncols.add(matrix.ncols)
    print(unique_ncols)

queries = load_queries()

nvals_infl(queries)
iter_number_infl(queries)
# unique_ncols(queries)
