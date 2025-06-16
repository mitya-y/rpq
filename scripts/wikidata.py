import matplotlib.pyplot as plt
from pprint import pprint
import statistics as stat
import numpy as np

QUERIES_NUM = 660

DATA_PATH = "wikidata"

# TODO: check errors - meadian and average value

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

for i in range(QUERIES_NUM):
    if results_cpu[i] == 0 or results_gpu[i] == 0:
        results_gpu[i], results_cpu[i] = 0, 0

total_gpu_time = sum(results_gpu)
total_cpu_time = sum(results_cpu)

print(total_cpu_time, total_gpu_time)

good_queries = [i for i in range(QUERIES_NUM) if results_gpu[i] != 0 and results_cpu[i] != 0]
original_indexes = [i + 1 for i in range(QUERIES_NUM) if results_gpu[i] != 0 and results_cpu[i] != 0]
good_cpu = [results_cpu[i] for i in good_queries]
good_gpu = [results_gpu[i] for i in good_queries]

for query in good_queries:
    if answers_cpu[query] != answers_gpu[query]:
        print(f"In query {query + 1} answers differ")

def worst_queries():
    worst = sorted([(q + 1, g / c, g, c, answers_cpu[q])
                   for q, c, g in zip(good_queries, good_cpu, good_gpu) if g / c > 1.5 and g > 1],
                   key=lambda x: x[1])
    worst.reverse()

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(8, 6),
        tight_layout=True
    )
    current_query_index = 0
    for ax1 in axes:
        # for ax in ax1:
            ax = ax1
            query = worst[current_query_index][0]
            current_query_index += 1
            i = good_queries.index(query - 1)
            num = 8
            queries = good_queries[max(i - num, 0) : min(i + num, len(good_queries) - 1)]

            x = original_indexes[max(i - num, 0) : min(i + num, len(original_indexes) - 1)]
            y = [answers_gpu[q] for q in queries]

            ax.plot(x, y, label="time(answers)")
            ax.set_xlabel('query number')
            ax.set_ylabel('answers number')
            ax.set_title(f"query {query}")

            for coord in x:
                ax.axvline(x=coord, color = 'red' if coord == query else 'green', alpha=0.3)
            ax.set_xticks(x)

            ax.legend()

    plt.tight_layout()
    plt.savefig('worst.svg')
    plt.show()
    return worst

def average_error():
    err_cpu = [e / v for e, v in zip(errors_cpu, results_cpu) if v != 0]
    err_gpu = [e / v for e, v in zip(errors_gpu, results_gpu) if v != 0]
    print(f"average error for CPU: {stat.mean(err_cpu)}, for GPU {stat.mean(err_gpu)}")
    print(f"median error for CPU: {stat.median(err_cpu)}, for GPU {stat.median(err_gpu)}")

def answers_dependency():
    res = [(results_gpu[q], answers_gpu[q]) for q in good_queries]
    res.sort(key=lambda x: x[1])
    x = [x[1] for x in res]
    y = [x[0] for x in res]

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label="time(answers)")
    plt.show()

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

def cloud():
    output='output.svg'
    s = 20
    colors = [
        # "#b25da6",
        "#ce4a4a",
        # "#eaaf41",
        "#6688c3",
        # "#48a56a",
        # "#769c9b",
    ]

    box_width = 0.6
    dataset_positions = []
    dataset_names = []
    for i, y_values, title in zip(range(2), [good_cpu, good_gpu], ["CPU", "GPU"]):
        color = colors[i % len(colors)]
        base_x = i
        x_values = []

        for _ in range(len(y_values)):
            drift = np.random.uniform(-0.15, 0.15)
            x_values.append(base_x + drift)

        plt.scatter(x_values, y_values,
                  color=color,
                  label=title,
                  alpha=0.8,
                  s=s)

        current_mean = np.mean(y_values)
        current_median = np.median(y_values)

        plt.hlines(y=current_mean, xmin=base_x-box_width/2, xmax=base_x+box_width/2,
                  colors=color, linestyles=':', linewidth=3, alpha=1.0)
        plt.hlines(y=current_median, xmin=base_x-box_width/2, xmax=base_x+box_width/2,
                  colors=color, linestyles='-', linewidth=3, alpha=1.0)

        dataset_positions.append(base_x)
        dataset_names.append(title)

        legend_elements = [
            plt.Line2D([0], [0], color='black', linestyle=':', linewidth=2, label='Mean'),
            plt.Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Median')
        ]
        plt.legend(handles=legend_elements, loc='lower right', fontsize=10)

        ax = plt.gca()
        ax.text(x=0.02+base_x*0.57,  # Отступ от левого края (1%)
            y=current_mean, 
            s=f"{current_mean:.3f}",  # Форматирование значения
            transform=ax.get_yaxis_transform(),  # Критично для позиционирования!
            verticalalignment='center',
            horizontalalignment='left',
            color=color,
            fontsize=10,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))  # Фон для читаемости

        ax.text(x=0.02+base_x*0.57,  # Отступ от левого края (1%)
            y=current_median,
            s=f"{current_median:.3f}",  # Форматирование значения
            transform=ax.get_yaxis_transform(),  # Критично для позиционирования!
            verticalalignment='center',
            horizontalalignment='left',
            color=color,
            fontsize=10,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))  # Фон для читаемости

    plt.xticks(dataset_positions, dataset_names)
    # ax.set_xticklabels(dataset_names, ha='right')

    plt.yscale('log')
    plt.ylabel('Execution time, s')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output)
    plt.show()

# answers_dependency()
# average_error()
# all_query_stats(True)

pprint(worst_queries())
# cloud()


