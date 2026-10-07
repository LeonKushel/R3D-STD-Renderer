#!/bin/zsh
# Rebuild libr3dcore.dylib from src/ (Apple Silicon, Xcode command-line tools).
set -e
cd "$(dirname "$0")"
swiftc -O -emit-library -o libr3dcore.dylib src/core.swift src/shaders.swift \
  -framework Metal -framework Foundation
echo "built: $(ls -la libr3dcore.dylib | awk '{print $5}') bytes"
