import numpy as np
from pprint import pprint

runs = 10
files = [open(f"result{i + 1}.txt") for i in range(runs)]

err = []

with open("result.txt", "w") as f:
    for lines in zip(*files):
        query = set()
        answer = set()
        time = []
        for line in lines:
            q, t, lt, a = line.split()
            query.add(int(q))
            answer.add(int(a))
            t = float(t) / 1_000_000
            time.append(float(t))

        if any([t == 0 for t in time]):
            continue

        if len(answer) != 1 or len(query) != 1:
            print(f"query {query} is bad")
        average_time = sum(time) / len(time)
        error = float(np.std(time, ddof=1))

        q = query.pop()
        f.write(f"{q} {average_time:.6f} {error:.6f} {answer.pop()}\n")

        err.append((q, error / average_time))
        print(f"{q}: {time}")

print("error stat:")
err.sort(key=lambda x: x[1])
err.reverse()
pprint(err[:10])
