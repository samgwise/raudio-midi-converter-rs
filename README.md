# CSV to MIDI Converter

A multi-interface Rust application for converting CSV audio analysis data to MIDI files, with both command-line and browser-based interfaces.

## Project Structure

This is a multi-crate workspace containing:

- **`csv-to-midi-core`** - Core library with CSV parsing and MIDI conversion logic
- **`csv-to-midi-cli`** - Command-line interface application
- **`csv-to-midi-wasm`** - WebAssembly bindings for browser usage
- **`www/`** - Web demo interface

## Features

- **Flexible CSV Parsing**: Handles audio analysis data with timestamps, frequencies, amplitudes, and optional channel indicators
- **MIDI Generation**: Converts frequency data to proper MIDI note numbers using equal temperament
- **Multi-Channel Support**: Supports up to 16 MIDI channels
- **Configurable Parameters**: Adjustable velocity, timing resolution, and note duration
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

## Installation & Usage

### Command Line Interface

1. **Build the project:**
   ```bash
   cargo build --release
   ```

2. **Run the CLI:**
   ```bash
   cargo run -p csv-to-midi-cli -- --input your-file.csv --output output.mid
   ```

**CLI Options:**
- `-i, --input <FILE>`: Input CSV file path
- `-o, --output <FILE>`: Output MIDI file path
- `-t, --ticks-per-quarter <NUMBER>`: MIDI ticks per quarter note (default: 480)
- `-v, --default-velocity <NUMBER>`: Default MIDI velocity 0-127 (default: 64)
- `-d, --min-duration <NUMBER>`: Minimum note duration in ticks (default: 100)

**Example:**
```bash
cargo run -p csv-to-midi-cli -- \
  --input bell-miner.csv \
  --output bell-miner.mid \
  --ticks-per-quarter 960 \
  --default-velocity 80
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

4. **Open your browser to:** `http://localhost:8000`

The web interface provides:
- Text area for pasting CSV data
- File upload for CSV files
- Real-time configuration options
- Instant MIDI file download

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
