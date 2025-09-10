#!/bin/bash
# Build script for WebAssembly module

set -e

echo "Building WASM module..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack is not installed. Please install it first:"
    echo "curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    exit 1
fi

# Build the WASM package
wasm-pack build csv-to-midi-wasm --target web --out-dir ../www/pkg

echo "WASM build complete! Generated files in www/pkg/"
echo ""
echo "To serve the web demo:"
echo "  cd www"
echo "  python -m http.server 8000"
echo "  # Then open http://localhost:8000"
