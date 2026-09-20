#!/usr/bin/env bash
# Compiler wrapper for Brian 2 standalone + OpenMP on macOS.
# Apple clang rejects a bare -fopenmp; it needs -Xpreprocessor -fopenmp plus
# Homebrew's libomp (brew install libomp). Use via: CXX=brain/cxx_omp.sh
args=()
for a in "$@"; do
  if [ "$a" = "-fopenmp" ]; then args+=("-Xpreprocessor" "-fopenmp"); else args+=("$a"); fi
done
exec c++ "${args[@]}"
