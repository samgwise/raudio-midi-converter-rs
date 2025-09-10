/**
 * Cloudflare Worker for CSV to MIDI Converter Web Application
 * Serves static files with proper MIME types, especially for WASM
 */

// MIME type mappings for different file extensions
const MIME_TYPES = {
  'html': 'text/html; charset=utf-8',
  'css': 'text/css',
  'js': 'application/javascript',
  'wasm': 'application/wasm',
  'json': 'application/json',
  'png': 'image/png',
  'jpg': 'image/jpeg',
  'jpeg': 'image/jpeg',
  'gif': 'image/gif',
  'svg': 'image/svg+xml',
  'ico': 'image/x-icon',
  'txt': 'text/plain',
  'md': 'text/markdown',
  'wav': 'audio/wav',
  'mp3': 'audio/mpeg',
  'midi': 'audio/midi',
  'mid': 'audio/midi',
  'csv': 'text/csv',
  'ts': 'application/typescript'
};

// Get MIME type from file extension
function getMimeType(filename) {
  const ext = filename.split('.').pop()?.toLowerCase();
  return MIME_TYPES[ext] || 'application/octet-stream';
}

// Add security headers
function addSecurityHeaders(response) {
  const headers = new Headers(response.headers);
  
  // Security headers
  headers.set('X-Content-Type-Options', 'nosniff');
  headers.set('X-Frame-Options', 'DENY');
  headers.set('X-XSS-Protection', '1; mode=block');
  headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');
  
  // CORS headers for WASM and other assets
  headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  
  // Cache headers for static assets
  if (response.headers.get('content-type')?.includes('wasm') || 
      response.headers.get('content-type')?.includes('javascript')) {
    headers.set('Cache-Control', 'public, max-age=86400'); // 24 hours
  } else {
    headers.set('Cache-Control', 'public, max-age=3600'); // 1 hour
  }
  
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: headers
  });
}

// Handle request routing
async function handleRequest(request) {
  const url = new URL(request.url);
  let pathname = url.pathname;
  
  // Handle root path
  if (pathname === '/') {
    pathname = '/index.html';
  }
  
  // Remove leading slash for asset lookup
  const assetPath = pathname.startsWith('/') ? pathname.slice(1) : pathname;
  
  try {
    // Try to get the asset from the static files
    const asset = await ASSETS.fetch(request.url);
    
    if (asset.status === 404) {
      // If asset not found, check for common routes
      if (pathname.includes('.')) {
        // File with extension not found
        return new Response('File not found', { status: 404 });
      } else {
        // SPA routing - serve index.html for paths without extensions
        const indexAsset = await ASSETS.fetch(new URL('/index.html', request.url).toString());
        if (indexAsset.status === 200) {
          const response = new Response(indexAsset.body, {
            ...indexAsset,
            headers: {
              ...indexAsset.headers,
              'Content-Type': 'text/html; charset=utf-8'
            }
          });
          return addSecurityHeaders(response);
        }
      }
      return new Response('Not Found', { status: 404 });
    }
    
    // Set correct MIME type based on file extension
    const filename = assetPath.split('/').pop() || '';
    const mimeType = getMimeType(filename);
    
    const response = new Response(asset.body, {
      status: asset.status,
      statusText: asset.statusText,
      headers: {
        ...asset.headers,
        'Content-Type': mimeType
      }
    });
    
    return addSecurityHeaders(response);
    
  } catch (error) {
    console.error('Error serving asset:', error);
    return new Response('Internal Server Error', { status: 500 });
  }
}

// Handle different HTTP methods
async function handleMethodRequest(request) {
  const { method } = request;
  
  switch (method) {
    case 'GET':
    case 'HEAD':
      return handleRequest(request);
    case 'OPTIONS':
      // Handle CORS preflight requests
      return new Response(null, {
        status: 200,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
          'Access-Control-Max-Age': '86400',
        }
      });
    default:
      return new Response('Method Not Allowed', { status: 405 });
  }
}

// Main event listener
addEventListener('fetch', event => {
  event.respondWith(handleMethodRequest(event.request));
});

// Handle worker lifecycle events
addEventListener('scheduled', event => {
  // Handle any scheduled tasks if needed
  event.waitUntil(handleScheduled(event));
});

async function handleScheduled(event) {
  // Placeholder for any scheduled maintenance tasks
  console.log('Scheduled event triggered:', event.cron);
}
