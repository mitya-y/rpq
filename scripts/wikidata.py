import matplotlib.pyplot as plt
from pprint import pprint

QUERIES_NUM = 660

DATA_PATH = "wikidata"

results_cpu = [0.0] * QUERIES_NUM
with open(f"data/cpu/{DATA_PATH}/all.txt") as file:
    for line in file:
        query, time = map(float, line.split())
        results_cpu[int(query) - 1] = time

results_gpu = [0.0] * QUERIES_NUM
with open(f"data/gpu/{DATA_PATH}/result.txt") as file:
    for line in file:
        query, time, _, _ = map(float, line.split())
        results_gpu[int(query) - 1] = time

for i in range(QUERIES_NUM):
    if results_cpu[i] == 0 or results_gpu[i] == 0:
        results_gpu[i], results_cpu[i] = 0, 0

total_cpu_time = sum(results_cpu)
total_gpu_time = sum(results_gpu)

print(total_cpu_time, total_gpu_time)

good_queries = [i for i in range(QUERIES_NUM) if results_gpu[i] != 0]
good_cpu = [results_cpu[i] for i in good_queries]
good_gpu = [results_gpu[i] for i in good_queries]

def worst_queries():
    return sorted([(q, g / c, g, c) for q, c, g in zip(good_queries, good_cpu, good_gpu) if g / c > 1.5 and g > 1], key=lambda x: x[1])


def all_query_stats(draw_together: bool = True):
    axis = list([g + 1 for g in good_queries])
    
    if draw_together:
        plt.figure(figsize=(8, 4))
        plt.plot(axis, good_gpu, label="GPU")
        plt.plot(axis, good_cpu, label="CPU")
        plt.xlabel('query number')
        plt.ylabel('time')
        plt.legend(fontsize=25)
    else:
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(8, 6),
            tight_layout=True
        )
        for ax, results, name in zip(axes, [good_cpu, good_gpu], ["CPU", "GPU"]):
            ax.plot(axis, results, label=name)
            ax.set_xlabel('query number')
            ax.set_ylabel('time')
            ax.set_title(name)
            ax.legend(title_fontsize=40)

    plt.show()
    

pprint(worst_queries())

all_query_stats(True)
