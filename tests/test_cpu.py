import psutil

n_workers = psutil.cpu_count(logical=False) - 1

print("n_workers: ", n_workers)