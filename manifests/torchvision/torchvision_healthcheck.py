import torch, torchvision

print("Torch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("TorchVision file:", torchvision.__file__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Arch list:", torch.cuda.get_arch_list())
print("Has NMS op:", hasattr(torch.ops.torchvision, "nms"))

boxes = torch.randn(1000, 4, device="cuda")
scores = torch.randn(1000, device="cuda")
keep = torchvision.ops.nms(boxes, scores, 0.5)
print("NMS device:", boxes.device)
print("NMS out:", keep.shape)
