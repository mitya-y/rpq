for i in range(1, 11):
    with open(f"result{i}.txt") as f:
        time = 0
        for line in f:
            time += float(line.split()[1])
        print(time)
