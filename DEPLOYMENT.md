# Cloudflare Workers Deployment Guide

This guide explains how to deploy the CSV to MIDI Converter web application to Cloudflare Workers.

## Prerequisites

1. **Cloudflare Account**: Sign up at [cloudflare.com](https://cloudflare.com)
2. **Node.js**: Version 18 or higher (required for Wrangler 4.x)
3. **Rust & wasm-pack**: For building WebAssembly modules
4. **Git**: For version control

## Initial Setup

### 1. Install Dependencies

```bash
# Install Node.js dependencies
npm install

# Install wasm-pack (if not already installed)
npm run wasm:install
# Or manually:
# curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

### 2. Configure Cloudflare Workers

1. **Get your Account ID**:
   - Go to [Cloudflare Dashboard](https://dash.cloudflare.com)
   - Copy your Account ID from the right sidebar
   - Edit `wrangler.toml` and uncomment the `account_id` line, replacing with your ID:
     ```toml
     account_id = "your-account-id-here"
     ```

2. **Authenticate with Cloudflare**:
   ```bash
   npx wrangler login
   ```

3. **Update Domain Configuration** (Optional):
   - If you have a custom domain, edit the `[routes]` section in `wrangler.toml`
   - Otherwise, remove the `[routes]` section to use the default `*.workers.dev` domain

### 3. Build the WebAssembly Module

```bash
# Build the WASM module for web deployment
npm run build:wasm
```

This command:
- Compiles the Rust code to WebAssembly
- Generates JavaScript bindings
- Outputs files to `www/pkg/`

## Deployment

### Development Deployment

```bash
# Start local development server
npm run dev
```

This starts a local server that mimics Cloudflare Workers environment.

### Staging Deployment

```bash
# Deploy to staging environment
npm run deploy:staging
```

### Production Deployment

```bash
# Deploy to production
npm run deploy:production

# Or use the default deployment (same as production)
npm run deploy
```

## Project Structure

```
csv-to-midi/
├── src/
│   └── worker.js          # Cloudflare Worker script
├── www/                   # Static web files
│   ├── index.html         # CSV processing interface
│   ├── audio-demo.html    # Audio processing interface
│   ├── pkg/               # Generated WASM files
│   └── README.md
├── csv-to-midi-wasm/      # Rust WASM crate
├── csv-to-midi-core/      # Core Rust library
├── wrangler.toml          # Cloudflare Workers config
├── package.json           # Node.js configuration
└── DEPLOYMENT.md          # This file
```

## Configuration Files

### wrangler.toml
Main Cloudflare Workers configuration:
- Worker name and main script
- Environment configurations (staging/production)
- Static asset configuration
- Custom domain routing (optional)

### package.json
Contains deployment scripts:
- `npm run build` - Builds WASM module
- `npm run deploy` - Deploys to Cloudflare Workers
- `npm run dev` - Local development server
- `npm run logs` - View worker logs

## Environment Management

The project supports multiple environments:

- **Development**: Local testing with `npm run dev`
- **Staging**: Testing environment with `npm run deploy:staging`
- **Production**: Live environment with `npm run deploy:production`

## Monitoring and Debugging

### View Logs
```bash
# View real-time logs for production
npm run logs

# View logs for staging
npm run logs:staging
```

### Analytics
- Visit the [Cloudflare Dashboard](https://dash.cloudflare.com)
- Navigate to Workers & Pages
- Select your worker to view analytics, logs, and performance metrics

## Troubleshooting

### Common Issues

1. **Wrangler Version Issues**:
   - If you get "unexpected fields" or "routes should be an array" errors, update Wrangler:
     ```bash
     npm install --save-dev wrangler@latest
     ```
   - Clear Wrangler cache if needed: `npx wrangler logout && npx wrangler login`

2. **WASM Loading Issues**:
   - Ensure MIME type `application/wasm` is properly set
   - Check that CORS headers are configured correctly
   - Verify the WASM file is accessible at the correct path

3. **Build Failures**:
   - Ensure Rust toolchain is installed: `rustup update`
   - Check wasm-pack is installed: `wasm-pack --version`
   - Try rebuilding: `npm run build:wasm`

4. **Deployment Errors**:
   - Verify account ID in `wrangler.toml`
   - Check authentication: `npx wrangler whoami`
   - Ensure you're within usage limits
   - For "ASSETS" binding errors, ensure using Wrangler 4.x

### Debug Steps

1. **Test Locally First**:
   ```bash
   npm run dev
   ```

2. **Check WASM Module**:
   ```bash
   # Verify WASM files exist
   ls www/pkg/
   
   # Should contain:
   # - csv_to_midi_wasm.js
   # - csv_to_midi_wasm_bg.wasm
   # - csv_to_midi_wasm.d.ts
   ```

3. **Preview Before Deployment**:
   ```bash
   npm run preview
   ```

## Custom Domain Setup

If you want to use a custom domain:

1. **Add Domain to Cloudflare**:
   - Add your domain to Cloudflare
   - Update DNS settings

2. **Configure Routes**:
   - Edit `wrangler.toml` routes section:
     ```toml
     [routes]
     pattern = "*your-domain.com/*"
     zone_name = "your-domain.com"
     ```

3. **Deploy**:
   ```bash
   npm run deploy
   ```

## Performance Optimization

The worker includes several optimizations:
- **Caching**: Static assets cached for 1-24 hours
- **Compression**: Automatic compression for text files
- **Security Headers**: CSP, CORS, and other security headers
- **MIME Types**: Proper MIME types for all assets, especially WASM

## Scaling Considerations

- **Free Tier**: 100,000 requests/day
- **Paid Plans**: Higher limits and additional features
- **Cold Starts**: First request may be slower
- **Memory Limits**: 128MB for free tier, 512MB for paid

## Security

The worker implements several security measures:
- Content Security Policy headers
- CORS headers for cross-origin requests
- XSS protection
- Frame options to prevent clickjacking

## Next Steps

After deployment:
1. Test all functionality (CSV and audio processing)
2. Set up monitoring and alerts
3. Configure custom domain (if needed)
4. Set up CI/CD pipeline for automated deployments

## Support

For issues specific to Cloudflare Workers:
- [Cloudflare Workers Documentation](https://developers.cloudflare.com/workers/)
- [Cloudflare Community](https://community.cloudflare.com/)

For application-specific issues:
- Check the main project README.md
- Review console logs in the browser
- Check Cloudflare Workers logs
