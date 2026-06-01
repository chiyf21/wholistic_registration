import time
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

while True:
    print("\033[2J\033[H", end="")  # 清屏
    for i in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(i)
            used = total - free
            name = torch.cuda.get_device_name(i)
            print(
                f"GPU {i}: {name}\n"
                f"  free: {free / 1024**3:.2f} GB\n"
                f"  used: {used / 1024**3:.2f} GB\n"
                f"  total: {total / 1024**3:.2f} GB\n"
            )
        except Exception as e:
            print(f"GPU {i}: failed to query memory: {e}")
    time.sleep(1)
