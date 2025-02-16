from pathlib import Path

QUERY_CNT = 660

total_load_time = 0
total_execute_time = 0

times = []

for query in range(1, QUERY_CNT + 1):
    filename = Path(f"queries_logs/{query}.txt")
    if filename.is_file():
        with open(filename, "r") as file:
            tmp = file.readline().split(',')
            load_time = float(tmp[0].split('=')[1])
            execute_time = float(tmp[1].split('=')[1])
            total_load_time += load_time
            total_execute_time += execute_time
            times.append((query, load_time, execute_time))

print(total_load_time)
print(total_execute_time)
# 30 and 255, in 255 no troubles

print('---------------------------------------')

divs = [(q, a / b) for (q, a, b) in times]
divs.sort(key=lambda t: t[1])
divs.reverse()

bads = [(q, t) for (q, t) in divs if t >= 1]
print("number of bads: ", len(bads))
print("number of all: ", len(times))

for i, d in enumerate(bads):
    if i > 10: break
    print(d)
# print(divs[:10])

print('---------------------------------------')
only_exec = [(query, ex) for (query, _, ex) in times]
only_exec.sort(key=lambda t: t[1])

for i, d in enumerate(only_exec):
    if i > 10: break
    print(d)

print('---------------------------------------')

import matplotlib.pyplot as plt

not_very_big_times = [(q, lt, et) for (q, lt, et) in times if et <= 0.1]
print("not big size: ", len(not_very_big_times))

x =  [q for (q, _, _) in not_very_big_times]
y1 = [lt for (_, lt, _) in not_very_big_times]
y2 = [et for (_, _, et) in not_very_big_times]

plt.figure(figsize=(8, 4))
plt.plot(x, y1, label="load")
plt.plot(x, y2, label="execute")
plt.legend()

plt.show()
