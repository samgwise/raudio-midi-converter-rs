# Quick Fix Script for Cloudflare Workers Deployment Issues
# Run this script if you encounter common deployment problems

Write-Host "🔧 Fixing common Cloudflare Workers deployment issues..." -ForegroundColor Green

# Update Wrangler to latest version
Write-Host "📦 Updating Wrangler to latest version..." -ForegroundColor Yellow
npm install --save-dev wrangler@latest

# Clear npm cache
Write-Host "🧹 Clearing npm cache..." -ForegroundColor Yellow
npm cache clean --force

# Check if wasm-pack is installed
Write-Host "🦀 Checking wasm-pack installation..." -ForegroundColor Yellow
try {
    $wasmPackVersion = wasm-pack --version
    Write-Host "✅ wasm-pack is installed: $wasmPackVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ wasm-pack not found. Please install it:" -ForegroundColor Red
    Write-Host "   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh" -ForegroundColor Cyan
}

# Build WASM module
Write-Host "🏗️  Building WASM module..." -ForegroundColor Yellow
try {
    npm run build:wasm
    Write-Host "✅ WASM build successful!" -ForegroundColor Green
} catch {
    Write-Host "❌ WASM build failed. Please check your Rust installation." -ForegroundColor Red
}

# Check Wrangler authentication
Write-Host "🔐 Checking Wrangler authentication..." -ForegroundColor Yellow
try {
    $wranglerUser = npx wrangler whoami 2>&1
    if ($wranglerUser -match "You are not authenticated") {
        Write-Host "❌ Not authenticated with Cloudflare. Please run:" -ForegroundColor Red
        Write-Host "   npx wrangler login" -ForegroundColor Cyan
    } else {
        Write-Host "✅ Wrangler authentication OK" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️  Could not check authentication status" -ForegroundColor Orange
}

# Validate wrangler.toml
Write-Host "📋 Validating wrangler.toml..." -ForegroundColor Yellow
if (Test-Path "wrangler.toml") {
    $tomlContent = Get-Content "wrangler.toml" -Raw
    
    # Check for common issues
    if ($tomlContent -match '\[\[assets\]\]') {
        Write-Host "❌ Old assets format detected. Please use [assets] instead of [[assets]]" -ForegroundColor Red
    }
    
    if ($tomlContent -match '\[routes\]' -and $tomlContent -notmatch 'routes\s*=\s*\[') {
        Write-Host "❌ Old routes format detected. Routes should be an array" -ForegroundColor Red
    }
    
    if ($tomlContent -match 'account_id.*=.*"your-account-id-here"') {
        Write-Host "⚠️  Please update your account_id in wrangler.toml" -ForegroundColor Orange
    }
    
    Write-Host "✅ wrangler.toml validation complete" -ForegroundColor Green
} else {
    Write-Host "❌ wrangler.toml not found!" -ForegroundColor Red
}

Write-Host "`n🎉 Fix script complete! Try deploying again with:" -ForegroundColor Green
Write-Host "   npm run deploy" -ForegroundColor Cyan
