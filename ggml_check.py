#!/usr/bin/env python3

"""
This script should work with any stock python3 interpreter.

This script receives a list of files on the commandline and will print for each the format/version.

GGML currently has 3 different formats:

        * `GGML` (unversioned): This was the original GGML version format, used for a variety of projects like whisper.cpp models
        * `GGMF`: This was the first release of the format that supported versioning
            * `v1` was the first and only release
        * `GGJT`: This is the current and most recent release of the format
            * `v1`: Added padding to the format
            * `v2`: Made breaking changes to the quantization format, requiring new sets of GGJT weights to be generated to run with
                    new versions of llama.cpp
            * `v3`: Made breaking changes to quantization format for Q4 and Q8

To see more, checkout the llama_file_version struct in llama.cpp of the llama.cpp project.
"""

import struct
import pathlib


GGML_FORMATS = {
    b"lmgg": "ggml",
    b"tjgg": "ggjt",
    b"fmgg": "ggmf",
}


def check_file(file: pathlib.Path):
    assert file.exists()
    assert file.is_file() or file.is_symlink()
    if file.is_symlink():
        file = file.readlink()
    with file.open(mode="rb") as fp:
        magic = fp.read(4)
        if len(magic) < 4:
            return "unknown"
        format = GGML_FORMATS.get(magic, "unknown")

        if format == "ggml":
            return "ggml (unversioned)"

        version_bytes = fp.read(4)
        if len(version_bytes) < 4:
            return "unknown"
        [version] = struct.unpack("<i", version_bytes)

        return f"{format} {version}"


if __name__ == "__main__":
    import sys

    files = sys.argv[1:]
    for ggml_file in files:
        version = check_file(pathlib.Path(ggml_file))
        print(f"{ggml_file}: {version}")
