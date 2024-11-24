import subprocess
out = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
mem = str(out.stdout.split('\n')[9]).split('|')[2].strip()
used = mem.split('/')[0].strip()
used = "".join([a for a in used if a.isnumeric()])
print(used)
