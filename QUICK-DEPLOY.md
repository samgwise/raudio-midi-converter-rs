# Quick Deployment Guide

This guide provides the fastest way to deploy your CSV-to-MIDI converter to Cloudflare Workers.

## 🚀 Quick Deploy (Recommended)

Since the WASM files are already built and committed, you can deploy immediately:

### 1. Set up Cloudflare Authentication
```bash
npx wrangler login
```

### 2. Add Your Account ID
Edit `wrangler.toml` and add your Cloudflare Account ID:
```toml
account_id = "your-account-id-here"
```
Get your Account ID from [Cloudflare Dashboard](https://dash.cloudflare.com/) (right sidebar).

### 3. Deploy Immediately
```bash
# Quick deploy (uses pre-built WASM files)
npm run deploy:quick
```

That's it! Your site should be live at `https://audio-to-midi.your-subdomain.workers.dev`

## 🔧 Alternative Deployment Methods

### Method 1: Standard Deploy
```bash
npm run deploy
```

### Method 2: Environment-Specific Deploy
```bash
# Deploy to staging
npm run deploy:staging

# Deploy to production  
npm run deploy:production
```

### Method 3: Rebuild WASM Then Deploy
If you've made changes to the Rust code:
```bash
# Rebuild WASM locally (requires wasm-pack installed)
npm run build:wasm

# Then deploy
npm run deploy
```

## ⚠️ Troubleshooting

### Problem: "wasm-pack not found" error
**Solution**: The WASM files are already built and committed. Use:
```bash
npm run deploy:quick
```

### Problem: "Unexpected fields found in assets"
**Solution**: Make sure you're using the updated `wrangler.toml` configuration.

### Problem: "Not authenticated"
**Solution**: 
```bash
npx wrangler login
```

### Problem: "Account ID not found"
**Solution**: Add your account ID to `wrangler.toml`:
1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. Copy your Account ID from the right sidebar
3. Add it to `wrangler.toml`

## 🌐 Custom Domain Setup

If you want to use your own domain instead of `workers.dev`:

1. **Add domain to Cloudflare**
2. **Update `wrangler.toml`**:
   ```toml
   routes = [
     { pattern = "*your-domain.com/*", zone_name = "your-domain.com" }
   ]
   ```
3. **Deploy**: `npm run deploy`

## 📊 Monitoring

After deployment, monitor your worker:
```bash
# View real-time logs
npm run logs

# View staging logs  
npm run logs:staging
```

## 🔄 Local Development

Test locally before deploying:
```bash
npm run dev
```
Then visit `http://localhost:8787`

---

**Need help?** See the full `DEPLOYMENT.md` for detailed instructions.
