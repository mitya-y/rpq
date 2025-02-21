import subprocess
COMMAND = "nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv"
out = subprocess.run(COMMAND.split(), capture_output=True, text=True)
mem_line = out.stdout.split('\n')[1]
all_mem, used_mem, avaible_mem = map(lambda s: int(s.split()[0]), mem_line.split(','))
print(used_mem)
