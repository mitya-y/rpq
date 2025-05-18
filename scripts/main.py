import numpy as np

RUNS = 1

def validate():
    expected = {}
    with open("data/result_cpu1.txt") as file:
        for line in file:
            query, _, result = map(int, line.split())
            expected[query] = result

    for i in range(1, RUNS + 1):
        with open(f"results/result{i}.txt") as file:
            for line in file:
                if "query" not in line or "skipped" in line:
                    continue
                result = int(line.split(':')[-1])
                query, _ = line.split(';')
                query = int(query.split('#')[1])
                if expected[query] != result:
                    print(query)

def convert():
    for i in range(1, RUNS + 1):
        file = open(f"results/converted{i}.txt", "w")
        with open(f"results/result{i}.txt") as f:
            for line in f:
                if "query" not in line or "skipped" in line:
                    continue
                query, time = line.split(';')
                query = int(query.split('#')[1])
                time = time.split(',')[1] # exec time
                time = float(time.split(':')[1])
                file.write(f"{query} {time}\n")
        file.close()

def collect():
    result_gpu = {}
    for i in range(1, RUNS + 1):
        with open(f"results/converted{i}.txt") as f:
            for line in f:
                query, time = map(float, line.split())
                query = int(query)
                if not query in result_gpu.keys():
                    result_gpu[query] = 0
                result_gpu[query] += time
    with open(f"results/result_gpu.txt", "w") as file:
        for query, time in result_gpu.items():
            file.write(f"{query} {time / RUNS}\n")

    result_cpu = {}
    for i in range(1, RUNS + 1):
        with open(f"results/result_cpu{i}.txt") as f:
            for line in f:
                query, time, _ = map(float, line.split())
                query = int(query)
                time /= 1_000_000
                if not query in result_cpu.keys():
                    result_cpu[query] = 0
                result_cpu[query] += time
    with open(f"results/result_cpu.txt", "w") as file:
        for query, time in result_cpu.items():
            file.write(f"{query} {time / RUNS}\n")

def calculate_errors():
    result_gpu = {}
    for i in range(1, RUNS + 1):
        with open(f"results/converted{i}.txt") as f:
            for line in f:
                query, time = map(float, line.split())
                query = int(query)
                if not query in result_gpu.keys():
                    result_gpu[query] = []
                result_gpu[query].append(time)
    errors_gpu = {}
    for query, res in result_gpu.items():
        errors_gpu[query] = np.std(res, ddof=1)

    result_cpu = {}
    for i in range(1, RUNS + 1):
        with open(f"results/result_cpu{i}.txt") as f:
            for line in f:
                query, time, _ = map(float, line.split())
                query = int(query)
                time /= 1_000_000
                if not query in result_cpu.keys():
                    result_cpu[query] = []
                result_cpu[query].append(time)
    errors_cpu = {}
    for query, res in result_cpu.items():
        errors_cpu[query] = np.std(res, ddof=1)

    file = open("results/errors.txt", "w")
    for i in range(670):
        if i in errors_cpu.keys() and i in errors_gpu.keys():
            file.write(f"{i} {errors_cpu[i]} {errors_gpu[i]}\n")
    file.close()

    file = open("results/abs_errors.txt", "w")
    for i in range(670):
        if i in errors_cpu.keys() and i in errors_gpu.keys():
            file.write(f"{i} {errors_cpu[i] / sum(result_cpu[i]) / RUNS} \
                        {errors_gpu[i] / sum(result_gpu[i]) / RUNS}\n")
    file.close()

def sort_errors():
    result = {}
    with open("results/abs_errors.txt") as file:
        for line in file:
            query, err_cpu, err_gpu = map(float, line.split())
            result[int(query)] = (err_cpu, err_gpu)

    errors_cpu = [cpu for cpu, _ in result.values()]
    errors_gpu = [gpu for _, gpu in result.values()]
    errors_cpu.sort()
    errors_gpu.sort()

    with open("results/sorted_cpu_errs.txt", "w") as file:
        for err in errors_cpu:
            file.write(f"{err}\n")

    with open("results/sorted_gpu_errs.txt", "w") as file:
        for err in errors_gpu:
            file.write(f"{err}\n")


def create_diff():
    cpu = {}
    gpu = {}

    with open("results/result_cpu.txt") as file:
        for line in file:
            n, time = map(float, line.split())
            cpu[int(n)] = time
    with open("results/result_gpu.txt") as file:
        for line in file:
            n, time = map(float, line.split())
            gpu[int(n)] = time

    file = open("results/bench_diff.txt", "w")
    file_big = open("results/bench_diff_big.txt", "w")
    for i in range(670):
        if i in cpu.keys() and i in gpu.keys():
            file.write(f"{i} {cpu[i]} {gpu[i]}\n")
            if cpu[i] > 1 or gpu[i] > 1:
                file_big.write(f"{i} {cpu[i]} {gpu[i]}\n")
    file.close()
    file_big.close()

def avg(a):
    return sum(b[1] for b in a) / len(a)

def load_errors():
    result = {}
    with open("results/errors.txt") as file:
        for line in file:
            query, err_cpu, err_gpu = map(float, line.split())
            result[int(query)] = (err_cpu, err_gpu)
    return result

def stat():
    results = []
    full_stat = {}
    with open("results/bench_diff.txt") as file:
        for line in file:
            query, time_cpu, time_gpu = map(float, line.split())
            query = int(query)
            full_stat[query] = (time_cpu, time_gpu)
            results.append((query, time_cpu / time_gpu))
    results.sort(key=lambda t: t[1])

    errors = load_errors()

    worst_querist = "".join([f"{a[0]} " for a in results[:5]])
    best_querist = "".join([f"{a[0]} " for a in results[-6:-1]])
    print(f"average: {avg(results)}")
    print(f"median: {results[len(results) // 2][1]}")
    print(f"worst 5 (queries: {worst_querist[:-1]}): {avg(results[:5])}")
    print(f"best 5 (queries: {best_querist[:-1]}): {avg(results[-6:-1])}")

    print("best:")
    for query, acsel in results[-6:-1]:
        time_cpu, time_gpu = full_stat[query]
        err_cpu, err_gpu = errors[query]
        err = abs(time_cpu * err_gpu - time_gpu * err_cpu) / (time_gpu ** 2);
        print(query, time_gpu, time_cpu, acsel, err)
    print("worst:")
    for query, acsel in results[:5]:
        time_cpu, time_gpu = full_stat[query]
        err_cpu, err_gpu = errors[query]
        err = abs(time_cpu * err_gpu - time_gpu * err_cpu) / (time_gpu ** 2);
        print(query, time_gpu, time_cpu, acsel, err)

def overall():
    time_cpu_all = 0
    time_gpu_all = 0
    wins = 0
    with open("results/bench_diff.txt") as file:
        for line in file:
            _, time_cpu, time_gpu = map(float, line.split())
            time_cpu_all += time_cpu
            time_gpu_all += time_gpu
            if time_gpu < time_cpu:
                wins += 1
    print(f"overall: {time_cpu_all / time_gpu_all}")
    print(f"wins: {wins}")

def sum_time():
    result_cpu = 0
    result_gpu = 0
    with open("results/bench_diff.txt") as file:
        for line in file:
            _, cpu, gpu = map(float, line.split())
            result_cpu += cpu
            result_gpu += gpu
    print(result_cpu)
    print(result_gpu)


# validate()
# convert()
# collect()
# create_diff()

# calculate_errors()
# sort_errors()

# stat()
# overall()
