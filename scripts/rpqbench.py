from os import access
import matplotlib.pyplot as plt

QUERIES_TYPES_NUM = 20
QUERIES_NUM_PER_TYPE = 1000

QUERIES_NUM = QUERIES_TYPES_NUM * QUERIES_NUM_PER_TYPE

DATA_PATH = "rpqbench-10kk"

results_cpu = [0.0] * QUERIES_NUM
answers_cpu = [0] * QUERIES_NUM
errors_cpu = [0.0] * QUERIES_NUM
with open(f"data/cpu/{DATA_PATH}/result.txt") as file:
    for line in file:
        query, time, error, ans = map(float, line.split())
        if query > QUERIES_NUM:
            continue
        results_cpu[int(query) - 1] = time
        errors_cpu[int(query) - 1] = error
        answers_cpu[int(query) - 1] = int(ans)

results_gpu = [0.0] * QUERIES_NUM
answers_gpu = [0.0] * QUERIES_NUM
errors_gpu = [0.0] * QUERIES_NUM
with open(f"data/gpu/{DATA_PATH}/result.txt") as file:
    for line in file:
        query, time, error, ans = map(float, line.split())
        if query > QUERIES_NUM:
            continue
        results_gpu[int(query) - 1] = time
        errors_gpu[int(query) - 1] = error
        answers_gpu[int(query) - 1] = int(ans)

total_cpu_time = sum(results_cpu)
total_gpu_time = sum(results_gpu)

print(total_cpu_time, total_gpu_time)

queries_type_cpu = [0.0] * QUERIES_TYPES_NUM
queries_type_gpu = [0.0] * QUERIES_TYPES_NUM
for i in range(QUERIES_TYPES_NUM):
    for j in range(i * QUERIES_NUM_PER_TYPE, (i + 1) * QUERIES_NUM_PER_TYPE):
        queries_type_cpu[i] += results_cpu[j]
        queries_type_gpu[i] += results_gpu[j]

def query_type_stats(queries_to_exclued = []):
    axis = list(range(1, QUERIES_TYPES_NUM + 1))
    axis = [axis[i] for i in range(QUERIES_TYPES_NUM) if i + 1 not in queries_to_exclued]
    cpu_y = [queries_type_cpu[i] for i in range(QUERIES_TYPES_NUM) if i + 1 not in queries_to_exclued]
    gpu_y = [queries_type_gpu[i] for i in range(QUERIES_TYPES_NUM) if i + 1 not in queries_to_exclued]

    fig, ax = plt.subplots()
    ax.plot(axis, cpu_y, label='CPU')
    ax.plot(axis, gpu_y, label='GPU')

    for x in axis:
        ax.axvline(x=x, color='green', alpha=0.3)
    ax.set_xticks(axis)
    ax.axhline(y=0, color='green', alpha=0.7, label='Ox')

    plt.xlabel('query type')
    plt.ylabel('time')
    plt.legend()
    plt.show()

    for i, c, g in zip(range(QUERIES_TYPES_NUM), cpu_y, gpu_y):
        print(f" & {g} & {c} & {g / c}")

def accelerates():
    for c, g, i in zip(queries_type_cpu, queries_type_gpu, range(QUERIES_NUM)):
        print(f"query {i + 1} accelerate is {g / c}")


def all_query_stats(draw_together: bool = True):
    axis = list(range(1, QUERIES_NUM + 1))

    if draw_together:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(axis, results_gpu, label="GPU")
        ax.plot(axis, results_cpu, label="CPU")

        ticks = [i * 1000 for i in range(1, QUERIES_TYPES_NUM)]
        ax.set_xticks(ticks)
        for x in ticks:
            ax.axvline(x=x, color='green', alpha=0.3)

        plt.xlabel('query number')
        plt.ylabel('time')
        plt.legend()
    else:
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(8, 6),
            tight_layout=True
        )
        for ax, results, name in zip(axes, [results_cpu, results_gpu], ["CPU", "GPU"]):
            ax.plot(axis, results, label=name)
            ax.set_xlabel('query number')
            ax.set_ylabel('time')
            ax.set_title(name)
            ticks = [i * 1000 for i in range(1, QUERIES_TYPES_NUM)]
            ax.set_xticks(ticks)
            for x in ticks:
                ax.axvline(x=x, color='green', alpha=0.3)
            ax.legend()

    plt.show()


def answers_dependency():
    res = [(r, a) for r, a in zip(results_gpu, answers_gpu)]
    res.sort(key=lambda x: x[1])
    x = [x[1] for x in res]
    y = [x[0] for x in res]

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label="time(answers)")
    plt.show()

def analize_iterration_number():
    path = "data/gpu/old/rpqbench-10kk/queries_logs"
    iters = [0] * QUERIES_NUM
    for i in range(QUERIES_NUM):
        with open(f"{path}/{i + 1}.txt") as f:
            # f.readline()
            iters[i] = int(f.readline().split('=')[1])
    x = [i + 1 for i in range(QUERIES_NUM)]
    plt.plot(x, iters, 'o', label="all types", markersize=1, color='blue')

    types = [11, 12]
    colors = ['red', 'green', 'yellow']
    for type, color in zip(types, colors):
        y = iters[(type - 1) * QUERIES_NUM_PER_TYPE : type * QUERIES_NUM_PER_TYPE]
        x = [i + 1 for i in range((type - 1) * QUERIES_NUM_PER_TYPE, type * QUERIES_NUM_PER_TYPE)]
        plt.plot(x, y, 'o', label=f"{type} type", markersize=2, color=color)

    ticks = [i * 1000 + 500 for i in range(QUERIES_TYPES_NUM)]
    labels = [f"{i + 1}" for i in range(QUERIES_TYPES_NUM)]
    plt.xticks(ticks, labels)
    bounds = [i * 1000 for i in range(QUERIES_TYPES_NUM + 1)]
    for x in bounds:
        plt.axvline(x=x, color='green', alpha=0.3)

    plt.xlabel('query type')
    plt.ylabel('iterrations number')
    plt.legend()
    plt.tight_layout()
    plt.savefig('itteration_number.svg')
    plt.show()

# query_type_stats()
# accelerates()
# all_query_stats(True)
analize_iterration_number()

# answers_dependency()

