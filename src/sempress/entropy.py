from __future__ import annotations
import zstandard as zstd

def zstd_compress(b: bytes, level: int = 10) -> bytes:
    c = zstd.ZstdCompressor(level=level)
    return c.compress(b)

def zstd_decompress(b: bytes) -> bytes:
    d = zstd.ZstdDecompressor()
    return d.decompress(b)
