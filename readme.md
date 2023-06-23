## GGML scripts

This repo is a collection of small Python scripts for use with [GGML](github.com/ggerganov/ggml) and [llama.cpp](github.com/ggerganov/llama.cpp) model files.

Scripts are intended to be:
  * Simple and focused, ideally should perform one well-documented function
  * Run with any modern python3 interpreter with zero dependencies

We might break these rules at some point when it's convenient but not right now.

### Scripts

- [`ggml_check.py`](./ggml_check.py): This script accepts on the command line a list of arguments as file paths. Has arguments for emitting complete information about a GGML file, including descriptions of all tensors and their types.
