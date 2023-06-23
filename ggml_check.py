#!/usr/bin/env python3

# This script should work with any stock python3 interpreter
from dataclasses import dataclass
from io import BufferedReader
import struct
import pathlib
import sys
from typing import Tuple, Union


GGML_FORMATS = {
    b"lmgg": "ggml",
    b"tjgg": "ggjt",
    b"fmgg": "ggmf",
}


@dataclass
class GGMLFileFields:
    filename: str
    fmt: str
    version: int | None
    n_vocab: int
    n_embd: int
    n_mult: int
    n_head: int
    n_layer: int
    n_rot: int
    ftype: str


LLAMA_FTYPES = [
    "LLAMA_FTYPE_ALL_F32",
    "LLAMA_FTYPE_MOSTLY_F16",
    "LLAMA_FTYPE_MOSTLY_Q4_0",
    "LLAMA_FTYPE_MOSTLY_Q4_1",
    "LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16",
    "LLAMA_FTYPE_MOSTLY_Q4_2",
    "LLAMA_FTYPE_MOSTLY_Q4_3",
    "LLAMA_FTYPE_MOSTLY_Q8_0",
    "LLAMA_FTYPE_MOSTLY_Q5_0",
    "LLAMA_FTYPE_MOSTLY_Q5_1",
    "LLAMA_FTYPE_MOSTLY_Q2_K",
    "LLAMA_FTYPE_MOSTLY_Q3_K_S",
    "LLAMA_FTYPE_MOSTLY_Q3_K_M",
    "LLAMA_FTYPE_MOSTLY_Q3_K_L",
    "LLAMA_FTYPE_MOSTLY_Q4_K_S",
    "LLAMA_FTYPE_MOSTLY_Q4_K_M",
    "LLAMA_FTYPE_MOSTLY_Q5_K_S",
    "LLAMA_FTYPE_MOSTLY_Q5_K_M",
    "LLAMA_FTYPE_MOSTLY_Q6_K",
]


class GGMLFile:
    def __init__(self, path: pathlib.Path) -> None:
        if path.is_symlink():
            path = path.readlink()
        assert path.exists() and path.is_file()
        self._path = path

    def read_structure(self) -> GGMLFileFields:
        with self._path.open("rb") as fp:
            fmt, version = self.read_magic(fp)
            n_vocab = self.read_u32(fp)
            n_embd = self.read_u32(fp)
            n_mult = self.read_u32(fp)
            n_head = self.read_u32(fp)
            n_layer = self.read_u32(fp)
            n_rot = self.read_u32(fp)
            ftype = LLAMA_FTYPES[self.read_u32(fp)]
            return GGMLFileFields(
                filename=str(self._path.absolute()),
                fmt=fmt,
                version=version,
                n_vocab=n_vocab,
                n_embd=n_embd,
                n_mult=n_mult,
                n_head=n_head,
                n_layer=n_layer,
                n_rot=n_rot,
                ftype=ftype,
            )

    def read_magic(self, fp: BufferedReader) -> Tuple[str, Union[int, None]]:
        magic = fp.read(4)
        if len(magic) < 4:
            return ("unknown", None)
        format = GGML_FORMATS.get(magic, "unknown")

        if format == "ggml":
            return ("ggml", None)

        version_bytes = fp.read(4)
        if len(version_bytes) < 4:
            return ("unknown", None)
        [version] = struct.unpack("<i", version_bytes)

        return (format, version)

    def read_u32(self, buf: BufferedReader) -> int:
        v = buf.read(4)
        [value] = struct.unpack("<i", v)
        return int(value)


if __name__ == "__main__":
    import argparse
    import json

    ggml_check = argparse.ArgumentParser("ggml_check")
    ggml_check.add_argument("filenames", nargs=argparse.ONE_OR_MORE)
    ggml_check.add_argument("-v", "--verbose", action="store_true")
    ggml_check.add_argument("-vv", "--verbose-pretty", action="store_true")
    ggml_check.add_argument("-s", "--suppress-failures", action="store_true")

    ggml_check_args = ggml_check.parse_args()

    for ggml_file in ggml_check_args.filenames:
        ggml = GGMLFile(pathlib.Path(ggml_file))
        try:
            ggml_struct = ggml.read_structure()
            if ggml_check_args.verbose:
                print(json.dumps(ggml_struct.__dict__))
            elif ggml_check_args.verbose_pretty:
                print(json.dumps(ggml_struct.__dict__, indent=4))
            else:
                print(f"{ggml_file}: {ggml_struct.fmt} {ggml_struct.version}")
        except Exception as e:
            if not ggml_check_args.suppress_failures:
                print(f"failed parsing {ggml_file}: {e}", file=sys.stderr)
