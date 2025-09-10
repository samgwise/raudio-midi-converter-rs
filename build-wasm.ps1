# Build script for WebAssembly module (Windows PowerShell)

Write-Host "Building WASM module..." -ForegroundColor Green

# Check if wasm-pack is installed
$wasmPack = Get-Command wasm-pack -ErrorAction SilentlyContinue
if (!$wasmPack) {
    Write-Host "wasm-pack is not installed. Please install it first:" -ForegroundColor Red
    Write-Host "Download from: https://rustwasm.github.io/wasm-pack/installer/" -ForegroundColor Yellow
    exit 1
}

try {
    # Build the WASM package
    wasm-pack build csv-to-midi-wasm --target web --out-dir ../www/pkg
    
    Write-Host "WASM build complete! Generated files in www/pkg/" -ForegroundColor Green
    Write-Host ""
    Write-Host "To serve the web demo:" -ForegroundColor Yellow
    Write-Host "  cd www" -ForegroundColor Cyan
    Write-Host "  python -m http.server 8000" -ForegroundColor Cyan
    Write-Host "  # Then open http://localhost:8000" -ForegroundColor Gray
} catch {
    Write-Host "Build failed: $_" -ForegroundColor Red
    exit 1
}
