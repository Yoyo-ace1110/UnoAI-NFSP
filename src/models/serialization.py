import io
import zlib
import torch

# 50MB 以上才壓縮
COMPRESS_THRESHOLD: int = 50 * 1024 * 1024
# zlib 壓縮檔的檔頭
_MAGIC: bytes = b"UNOZ1"

def save_maybe_compressed(
    obj: dict[str, object], 
    path: str, 
    threshold: int = COMPRESS_THRESHOLD
) -> tuple[int, bool]:
    """ 存檔: 序列化後大於 threshold 則使用 zlib 壓縮, 回傳 (位元組數, 是否壓縮) """
    buffer: io.BytesIO = io.BytesIO()
    torch.save(obj, buffer)
    raw: bytes = buffer.getvalue()
    if (len(raw) <= threshold):
        with open(path, "wb") as file: file.write(raw)
        return len(raw), False
    # 分段壓縮直接寫檔, 避免同時在記憶體裡放一份完整的壓縮副本
    compressor = zlib.compressobj(level=3)
    written: int = len(_MAGIC)
    with open(path, "wb") as file:
        file.write(_MAGIC)
        view: memoryview = memoryview(raw)
        for start in range(0, len(raw), 1 << 24):
            chunk: bytes = compressor.compress(view[start:start + (1 << 24)])
            file.write(chunk)
            written += len(chunk)
        tail: bytes = compressor.flush()
        file.write(tail)
        written += len(tail)
    return written, True

def load_maybe_compressed(path: str): 
    """ 讀檔: 自動辨識壓縮檔與一般檔 (舊版存的檔案也可以直接讀) """
    with open(path, "rb") as file: head: bytes = file.read(len(_MAGIC))
    if (head == _MAGIC):
        with open(path, "rb") as file:
            file.seek(len(_MAGIC))
            data: bytes = zlib.decompress(file.read())
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)  # type: ignore
    return torch.load(path, map_location="cpu", weights_only=True)  # type: ignore
