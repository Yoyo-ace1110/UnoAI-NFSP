import torch
from typing import Any

# 自動偵測是否具備 NVIDIA CUDA，否則使用 CPU (無需支援 Apple MPS)
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(*items: Any) -> Any:
    """ 將傳入的一或多個 Tensor 移至當前裝置 """
    result = tuple(item.to(DEVICE) if isinstance(item, torch.Tensor) else item for item in items)
    return (result[0] if len(result) == 1 else result)

def get_device_info() -> str:
    """ 取得當前裝置的詳細資訊 """
    if (DEVICE.type == "cuda"):
        return f"使用裝置: GPU ({torch.cuda.get_device_name(0)})"
    return "使用裝置: CPU"

def print_device_info() -> None:
    """ 輸出目前使用的裝置 """
    print(get_device_info())
