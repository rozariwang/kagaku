import torch

# This will print True if CUDA is available, otherwise False
print(torch.cuda.is_available())

# Optionally, print out the number of CUDA devices and their names
if torch.cuda.is_available():
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"CUDA Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")