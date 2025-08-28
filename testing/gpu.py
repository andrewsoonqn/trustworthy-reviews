import torch

if torch.cuda.is_available():
    print("GPU is available!")
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("No GPU detected. Running on CPU.")

