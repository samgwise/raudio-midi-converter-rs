# CSV to MIDI Web Interface

This directory contains the web-based interface for the CSV to MIDI converter.

## Files

- **`index.html`** - Original CSV-only processing interface
- **`audio-demo.html`** - Enhanced interface with both CSV and audio processing capabilities
- **`pkg/`** - Generated WebAssembly module and JavaScript bindings
- **`README.md`** - This file

## Features

### CSV Processing Mode
- Paste CSV data directly or upload CSV files
- Real-time MIDI conversion in the browser
- Configurable MIDI parameters (velocity, timing, etc.)
- Instant download of generated MIDI files

### Audio Processing Mode (audio-demo.html only)
- Upload audio files (WAV, MP3, M4A, etc.)
- Real-time audio waveform visualization
- Configurable pitch detection parameters:
  - Frequency range (min/max Hz)
  - Detection threshold
  - Frame size and hop size
- Browser-based pitch detection using autocorrelation
- Direct conversion from audio to MIDI
- **Enhanced MIDI output with CC data:**
  - **CC 100**: Pitch contour (fine pitch variations)
  - **CC 101**: Amplitude (real-time volume changes)

## Usage

1. **Start a local web server** in this directory:
   ```bash
   # Using Python
   python -m http.server 8000
   
   # Using Node.js
   npx serve .
   
   # Using PHP
   php -S localhost:8000
   ```

2. **Open your browser** and navigate to:
   - `http://localhost:8000/index.html` - CSV-only interface
   - `http://localhost:8000/audio-demo.html` - Full audio demo interface

## Browser Compatibility

- **CSV Processing**: Works in all modern browsers
- **Audio Processing**: Requires browsers with Web Audio API support:
  - Chrome 36+
  - Firefox 25+
  - Safari 14.1+
  - Edge 79+

## Audio Format Support

The audio processing mode supports any audio format that the browser can decode:
- **WAV** - Uncompressed audio (best quality)
- **MP3** - Compressed audio (widely supported)
- **M4A/AAC** - Compressed audio (good quality)
- **OGG** - Open source format (Firefox, Chrome)
- **FLAC** - Lossless compression (limited browser support)

## Technical Details

### Audio Processing Pipeline
1. **File Upload**: Audio file is loaded using FileReader API
2. **Decoding**: Browser's Web Audio API decodes the audio data
3. **Visualization**: Waveform is drawn on HTML5 Canvas
4. **Analysis**: WebAssembly module processes audio samples:
   - Frame-based analysis with configurable window sizes
   - Autocorrelation-based pitch detection algorithm
   - Frequency range filtering and threshold-based detection
5. **Conversion**: Detected pitches are converted to MIDI events with CC data:
   - MIDI notes for detected pitches
   - CC 100 for pitch contour (microtonal variations)
   - CC 101 for amplitude envelope
6. **Download**: Generated MIDI file is available for download

### Performance Notes
- Audio processing runs entirely in the browser (no server required)
- WebAssembly provides near-native performance for pitch detection
- Large audio files may take several seconds to process
- Processing time depends on audio length, frame size, and hop size settings

## Configuration

### Audio Analysis Parameters
- **Min/Max Frequency**: Define the frequency range of interest
  - Default: 65-2093 Hz (covers most musical instruments)
  - Vocal range: 80-800 Hz
  - Piano range: 27.5-4186 Hz
- **Threshold**: Controls sensitivity of pitch detection
  - Higher values: Fewer false positives, may miss quiet notes
  - Lower values: More sensitive, may include noise
- **Frame Size**: Analysis window size
  - Larger frames: Better frequency resolution, worse time resolution
  - Smaller frames: Better time resolution, worse frequency resolution
- **Hop Size**: Step size between analysis frames
  - Smaller hop sizes: Better time resolution, more processing time

### MIDI Parameters
- **Ticks per Quarter**: MIDI timing resolution (typically 480)
- **Default Velocity**: Note loudness (0-127)
- **Min Note Duration**: Prevents very short notes
- **Base Octave**: Reference octave for frequency conversion

## Limitations

- Audio processing is CPU-intensive and may be slow on older devices
- Very long audio files (>10 minutes) may cause browser memory issues
- Pitch detection works best with monophonic (single note) content
- Complex polyphonic music may not be accurately converted

## Development

The WebAssembly module is built from Rust source code in the `csv-to-midi-wasm` crate. To rebuild:

```bash
wasm-pack build csv-to-midi-wasm --target web --out-dir ../www/pkg
```
