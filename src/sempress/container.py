from __future__ import annotations
import msgpack
from .entropy import zstd_compress, zstd_decompress

MAGIC = b"SEMZ1\x00"

def pack_container(payload: dict) -> bytes:
    blob = msgpack.packb(payload, use_bin_type=True)
    comp = zstd_compress(blob)
    return MAGIC + comp

def unpack_container(b: bytes) -> dict:
    assert b.startswith(MAGIC), "Not a SEMZ container or wrong version"
    comp = b[len(MAGIC):]
    raw = zstd_decompress(comp)
    return msgpack.unpackb(raw, raw=False)
