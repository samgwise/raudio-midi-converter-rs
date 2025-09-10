# CSV to MIDI Converter

A multi-interface Rust application for converting CSV audio analysis data and audio files to MIDI files, with both command-line and browser-based interfaces.

## Project Structure

This is a multi-crate workspace containing:

- **`csv-to-midi-core`** - Core library with CSV parsing and MIDI conversion logic
- **`csv-to-midi-cli`** - Command-line interface application
- **`csv-to-midi-wasm`** - WebAssembly bindings for browser usage
- **`www/`** - Web demo interface

## Features

- **Flexible Input Support**: 
  - CSV audio analysis data with timestamps, frequencies, amplitudes, and optional channel indicators
  - Direct audio file processing (WAV and FLAC formats) with pitch detection
- **Advanced Audio Analysis**: Real-time pitch detection using autocorrelation-based algorithms
- **MIDI Generation**: Converts frequency data to proper MIDI note numbers using equal temperament
- **Multi-Channel Support**: Supports up to 16 MIDI channels
- **Configurable Parameters**: Adjustable velocity, timing resolution, note duration, and audio analysis parameters
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Browser Compatible**: Runs in modern web browsers via WebAssembly

## CSV Format

The expected CSV format is:
```
timestamp,frequency,amplitude,[channel]
```

**Example:**
```csv
0.241904762,440.0,0.5,[1]
0.5,880.0,0.7,[2]
1.0,0,0.0
```

- **timestamp**: Time in seconds
- **frequency**: Frequency in Hz (0 indicates silence)
- **amplitude**: Amplitude from 0.0 to 1.0
- **[channel]**: Optional MIDI channel (1 or 2), leave empty for silence

## Audio Format Support

When built with the `audio` feature, the application supports direct processing of audio files:

### Supported Formats
- **WAV**: Uncompressed audio files (.wav)
- **FLAC**: Free Lossless Audio Codec files (.flac)

### Audio Processing Pipeline
1. **Audio Loading**: Reads audio file and extracts PCM samples
2. **Resampling**: Automatically resamples to target sample rate if needed
3. **Frame Analysis**: Processes audio in overlapping frames using configurable window sizes
4. **Pitch Detection**: Uses autocorrelation-based algorithm to estimate fundamental frequency
5. **Event Generation**: Converts pitch detections to time-stamped audio events
6. **MIDI Conversion**: Maps frequencies to MIDI notes using standard equal temperament

### Audio Analysis Parameters
- **Frame Size**: Larger frames provide better frequency resolution but worse time resolution
- **Hop Size**: Smaller hop sizes provide better time resolution but increase processing time
- **Frequency Range**: Define the frequency range of interest (e.g., vocal range: 80-800 Hz)
- **Threshold**: Higher thresholds reduce false positives but may miss quiet notes

### Web Audio Demo

The `www/audio-demo.html` interface provides a complete browser-based audio processing experience:
- **Upload any audio format** supported by your browser (WAV, MP3, M4A, etc.)
- **Real-time waveform visualization** shows your audio data
- **Interactive controls** for all pitch detection parameters
- **Instant processing** using WebAssembly for near-native performance
- **No installation required** - everything runs in your browser

This makes it easy to experiment with different audio files and settings without needing to install the CLI tools.

## Installation & Usage

### Command Line Interface

1. **Build the project:**
   ```bash
   # For CSV-only functionality:
   cargo build --release
   
   # For audio file processing support:
   cargo build --release --features audio
   ```

2. **Run the CLI:**
   ```bash
   # Process CSV file:
   cargo run -p csv-to-midi-cli -- --input your-file.csv --output output.mid
   
   # Process audio file (requires --features audio):
   cargo run -p csv-to-midi-cli --features audio -- --input audio.wav --output output.mid
   ```

**CLI Options:**
- `-i, --input <FILE>`: Input file path (CSV or audio file: .wav, .flac)
- `-o, --output <FILE>`: Output MIDI file path
- `-t, --ticks-per-quarter <NUMBER>`: MIDI ticks per quarter note (default: 480)
- `-v, --default-velocity <NUMBER>`: Default MIDI velocity 0-127 (default: 64)
- `-d, --min-duration <NUMBER>`: Minimum note duration in ticks (default: 100)

**Audio Analysis Options (when using --features audio):**
- `--fmin <HZ>`: Minimum frequency for analysis in Hz (default: 65.0)
- `--fmax <HZ>`: Maximum frequency for analysis in Hz (default: 2093.0)
- `--threshold <NUMBER>`: Threshold for voiced/unvoiced detection 0.0-1.0 (default: 0.1)
- `--frame-size <SAMPLES>`: Frame size for analysis in samples (default: 2048)
- `--hop-size <SAMPLES>`: Hop size for analysis in samples (default: 512)

**Examples:**

*CSV Processing:*
```bash
cargo run -p csv-to-midi-cli -- \
  --input bell-miner.csv \
  --output bell-miner.mid \
  --ticks-per-quarter 960 \
  --default-velocity 80
```

*Audio Processing:*
```bash
cargo run -p csv-to-midi-cli --features audio -- \
  --input audio-sample.wav \
  --output audio-to-midi.mid \
  --fmin 100.0 \
  --fmax 2000.0 \
  --threshold 0.3 \
  --frame-size 4096 \
  --hop-size 1024
```

### Web Interface

The web interface requires `wasm-pack` to build the WebAssembly module.

1. **Install wasm-pack:**
   ```bash
   curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
   # OR on Windows:
   # Download from https://rustwasm.github.io/wasm-pack/installer/
   ```

2. **Build the WASM module:**
   ```bash
   wasm-pack build csv-to-midi-wasm --target web --out-dir ../www/pkg
   ```

3. **Serve the web demo:**
   ```bash
   cd www
   python -m http.server 8000
   # OR use any other local server
   ```

4. **Open your browser to:**
   - `http://localhost:8000/index.html` - CSV processing interface
   - `http://localhost:8000/audio-demo.html` - Enhanced interface with audio processing

The web interface provides:
- **CSV Processing**: Text area for pasting CSV data, file upload, real-time configuration
- **Audio Processing**: Upload audio files (WAV, MP3, etc.), pitch detection, waveform visualization
- **Real-time conversion**: Instant MIDI file download for both modes
- **Browser-based**: No server-side processing required

## Configuration Options

### ConversionConfig

- **`ticks_per_quarter`** (u16): MIDI timing resolution (default: 480)
- **`min_note_duration`** (u32): Minimum note duration in ticks (default: 100)
- **`default_velocity`** (u8): Default note velocity 0-127 (default: 64)
- **`base_octave`** (u8): Base octave for frequency conversion (default: 4)

## Development

### Running Tests

```bash
# Test core library
cargo test -p csv-to-midi-core

# Test all crates
cargo test

# Run with output
cargo test -- --nocapture
```

### Building for Release

```bash
# Build CLI
cargo build --release -p csv-to-midi-cli

# Build WASM
wasm-pack build csv-to-midi-wasm --target web --release
```

## Technical Details

### Frequency to MIDI Note Conversion

Uses the standard equal temperament formula:
```
MIDI_note = 69 + 12 * log2(frequency / 440)
```

Where A4 = 440Hz = MIDI note 69.

### Amplitude to Velocity Mapping

Linearly maps amplitude (0.0-1.0) to MIDI velocity (0-127) with a minimum threshold to ensure audible notes.

### Channel Assignment

- Channel indicators `[1]` and `[2]` in CSV map to MIDI channels 0 and 1 respectively
- Silence events (frequency = 0) end all active notes
- Multiple simultaneous notes per channel are supported

### MIDI File Structure

Generates standard MIDI format with:
- Format 1 (multi-track)
- Configurable timing resolution
- Separate tracks for different channels
- Proper note-on/note-off event timing

## Dependencies

### Core Dependencies
- `csv` - CSV parsing
- `midly` - MIDI file generation
- `serde` - Serialization support
- `thiserror` - Error handling

### Audio Dependencies (optional, enabled with `audio` feature)
- `hound` - WAV file reading
- `rubato` - Audio resampling
- `claxon` - FLAC file decoding
- Built-in autocorrelation-based pitch detection

### CLI Dependencies
- `clap` - Command-line argument parsing
- `anyhow` - Error handling

### WASM Dependencies
- `wasm-bindgen` - JavaScript bindings
- `js-sys` - JavaScript API bindings
- `web-sys` - Web API bindings
- `console_error_panic_hook` - Better error reporting
- `wee_alloc` - Optimized allocator

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `cargo test`
5. Submit a pull request

## License

This project is licensed under the MIT OR Apache-2.0 license.

## Example Output

Converting the provided `bell-miner.csv` file generates a MIDI file with:
- Two channels of audio data
- High-frequency content (2793.83 Hz) mapped to high MIDI notes
- Low-frequency content mapped to lower notes
- Silence periods properly handled
- Total duration based on timestamp ranges

The resulting MIDI file can be played in any standard MIDI player or imported into digital audio workstations.
