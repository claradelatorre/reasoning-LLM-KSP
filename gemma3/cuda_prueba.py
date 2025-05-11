import torch

print("PyTorch detecta CUDA:", torch.cuda.is_available())
print("Número de GPUs disponibles:", torch.cuda.device_count())
print("Nombre de la GPU activa:", torch.cuda.get_device_name(0))
print("Dispositivo por defecto:", torch.cuda.current_device())
