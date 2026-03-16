import struct
from pathlib import Path
import numpy as np


def read_dmb_file(file_path: Path, is_confidence: bool = False):
    """
    Read DMB file used by legacy depth/confidence pipelines.
    """
    try:
        with open(file_path, "rb") as f:
            type_val = struct.unpack("<i", f.read(4))[0]
            h = struct.unpack("<i", f.read(4))[0]
            w = struct.unpack("<i", f.read(4))[0]
            nb = struct.unpack("<i", f.read(4))[0]

            if type_val != 1:
                raise ValueError(f"Unsupported DMB type: {type_val}")

            data_size = h * w * nb
            if is_confidence:
                raw = f.read(data_size)
                data_array = np.frombuffer(raw, dtype=np.uint8, count=data_size)
                data_array = data_array.reshape(h, w, nb).astype(np.float32)
            else:
                raw = f.read(data_size * 4)
                data_array = np.frombuffer(raw, dtype=np.float32, count=data_size)
                data_array = data_array.reshape(h, w, nb)

            if nb == 1:
                data_array = data_array.squeeze()
            return data_array
    except Exception as e:
        raise RuntimeError(f"Failed to read DMB file {file_path}: {e}") from e
