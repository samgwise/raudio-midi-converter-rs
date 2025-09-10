use csv_to_midi_core::{convert_csv_string_to_midi, ConversionConfig, AudioEvent, convert_to_midi_events_with_cc, generate_midi_file_with_cc};
use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// Configuration for CSV to MIDI conversion (WASM-exposed version)
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct WasmConversionConfig {
    ticks_per_quarter: u16,
    min_note_duration: u32,
    default_velocity: u8,
    base_octave: u8,
}

#[wasm_bindgen]
impl WasmConversionConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(
        ticks_per_quarter: Option<u16>,
        min_note_duration: Option<u32>,
        default_velocity: Option<u8>,
        base_octave: Option<u8>,
    ) -> WasmConversionConfig {
        WasmConversionConfig {
            ticks_per_quarter: ticks_per_quarter.unwrap_or(480),
            min_note_duration: min_note_duration.unwrap_or(100),
            default_velocity: default_velocity.unwrap_or(64),
            base_octave: base_octave.unwrap_or(4),
        }
    }
    
    #[wasm_bindgen(getter)]
    pub fn ticks_per_quarter(&self) -> u16 {
        self.ticks_per_quarter
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_ticks_per_quarter(&mut self, value: u16) {
        self.ticks_per_quarter = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn min_note_duration(&self) -> u32 {
        self.min_note_duration
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_min_note_duration(&mut self, value: u32) {
        self.min_note_duration = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn default_velocity(&self) -> u8 {
        self.default_velocity
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_default_velocity(&mut self, value: u8) {
        self.default_velocity = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn base_octave(&self) -> u8 {
        self.base_octave
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_base_octave(&mut self, value: u8) {
        self.base_octave = value;
    }
}

impl From<WasmConversionConfig> for ConversionConfig {
    fn from(config: WasmConversionConfig) -> Self {
        ConversionConfig {
            ticks_per_quarter: config.ticks_per_quarter,
            min_note_duration: config.min_note_duration,
            default_velocity: config.default_velocity,
            base_octave: config.base_octave,
        }
    }
}

/// Convert CSV string to MIDI byte array
/// 
/// # Arguments
/// * `csv_data` - CSV string containing audio analysis data
/// * `config` - Configuration object for conversion options
/// 
/// # Returns
/// * Uint8Array containing MIDI file data, or throws error if conversion fails
#[wasm_bindgen]
pub fn convert_csv_to_midi_wasm(
    csv_data: &str,
    config: &WasmConversionConfig,
) -> Result<js_sys::Uint8Array, JsValue> {
    console_log!("Starting CSV to MIDI conversion...");
    console_log!("CSV data length: {} bytes", csv_data.len());
    
    let core_config: ConversionConfig = config.clone().into();
    
    let midi_data = convert_csv_string_to_midi(csv_data, core_config)
        .map_err(|e| JsValue::from_str(&format!("Conversion error: {}", e)))?;
    
    console_log!("Conversion successful! Generated {} bytes of MIDI data", midi_data.len());
    
    Ok(js_sys::Uint8Array::from(&midi_data[..]))
}

/// Create a default configuration
#[wasm_bindgen]
pub fn create_default_config() -> WasmConversionConfig {
    WasmConversionConfig::new(None, None, None, None)
}

/// Audio analysis configuration for web interface
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct WasmAudioConfig {
    sample_rate: f32,
    frame_size: usize,
    hop_size: usize,
    min_frequency: f32,
    max_frequency: f32,
    threshold: f32,
}

#[wasm_bindgen]
impl WasmAudioConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(
        sample_rate: Option<f32>,
        frame_size: Option<usize>,
        hop_size: Option<usize>,
        min_frequency: Option<f32>,
        max_frequency: Option<f32>,
        threshold: Option<f32>,
    ) -> WasmAudioConfig {
        WasmAudioConfig {
            sample_rate: sample_rate.unwrap_or(44100.0),
            frame_size: frame_size.unwrap_or(2048),
            hop_size: hop_size.unwrap_or(512),
            min_frequency: min_frequency.unwrap_or(65.0),
            max_frequency: max_frequency.unwrap_or(2093.0),
            threshold: threshold.unwrap_or(0.1),
        }
    }
    
    #[wasm_bindgen(getter)]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_sample_rate(&mut self, value: f32) {
        self.sample_rate = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_frame_size(&mut self, value: usize) {
        self.frame_size = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_hop_size(&mut self, value: usize) {
        self.hop_size = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn min_frequency(&self) -> f32 {
        self.min_frequency
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_min_frequency(&mut self, value: f32) {
        self.min_frequency = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn max_frequency(&self) -> f32 {
        self.max_frequency
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_max_frequency(&mut self, value: f32) {
        self.max_frequency = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_threshold(&mut self, value: f32) {
        self.threshold = value;
    }
}

/// Process audio samples and convert to MIDI
/// 
/// # Arguments
/// * `samples` - Float32Array of audio samples
/// * `audio_config` - Audio analysis configuration
/// * `midi_config` - MIDI conversion configuration
/// 
/// # Returns
/// * Uint8Array containing MIDI file data
#[wasm_bindgen]
pub fn process_audio_to_midi(
    samples: &[f32],
    audio_config: &WasmAudioConfig,
    midi_config: &WasmConversionConfig,
) -> Result<js_sys::Uint8Array, JsValue> {
    console_log!("Starting audio processing...");
    console_log!("Audio samples: {} at {}Hz", samples.len(), audio_config.sample_rate);
    
    let audio_events = analyze_audio_samples(
        samples,
        audio_config.sample_rate,
        audio_config.frame_size,
        audio_config.hop_size,
        audio_config.min_frequency,
        audio_config.max_frequency,
        audio_config.threshold,
    )?;
    
    console_log!("Generated {} audio events", audio_events.len());
    
    let core_config: ConversionConfig = midi_config.clone().into();
    
    // Convert audio events to MIDI events with CC data
    let midi_collection = convert_to_midi_events_with_cc(&audio_events, &core_config)
        .map_err(|e| JsValue::from_str(&format!("MIDI event conversion error: {}", e)))?;
    
    // Generate MIDI file with CC events for pitch contour (CC100) and amplitude (CC101)
    let midi_data = generate_midi_file_with_cc(midi_collection, &core_config)
        .map_err(|e| JsValue::from_str(&format!("MIDI file generation error: {}", e)))?;
    
    console_log!("Audio to MIDI conversion successful! Generated {} bytes", midi_data.len());
    
    Ok(js_sys::Uint8Array::from(&midi_data[..]))
}

/// Analyze audio samples using autocorrelation-based pitch detection
fn analyze_audio_samples(
    samples: &[f32],
    sample_rate: f32,
    frame_size: usize,
    hop_size: usize,
    fmin: f32,
    fmax: f32,
    threshold: f32,
) -> Result<Vec<AudioEvent>, JsValue> {
    let mut events = Vec::new();
    
    // Process audio in overlapping frames
    for frame_start in (0..samples.len()).step_by(hop_size) {
        if frame_start + frame_size > samples.len() {
            break;
        }
        
        let frame = &samples[frame_start..frame_start + frame_size];
        let time = frame_start as f64 / sample_rate as f64;
        
        // Simple autocorrelation-based pitch detection
        let (frequency, confidence) = estimate_pitch_autocorr(frame, sample_rate, fmin, fmax)?;
        let voiced = frequency > 0.0 && confidence > threshold;
        
        if voiced {
            let amplitude = calculate_rms_amplitude(frame);
            events.push(AudioEvent {
                line_number: (frame_start / hop_size) as u32 + 1,
                timestamp: time,
                frequency,
                amplitude: amplitude as f64,
                channel: Some(1),
            });
        }
    }
    
    Ok(events)
}

/// Simple autocorrelation-based pitch estimation
fn estimate_pitch_autocorr(
    frame: &[f32],
    sample_rate: f32,
    fmin: f32,
    fmax: f32,
) -> Result<(f64, f32), JsValue> {
    if frame.len() < 2 {
        return Ok((0.0, 0.0));
    }
    
    // Calculate autocorrelation
    let max_lag = ((sample_rate / fmin).min(frame.len() as f32 / 2.0)) as usize;
    let min_lag = ((sample_rate / fmax).max(1.0)) as usize;
    
    if min_lag >= max_lag {
        return Ok((0.0, 0.0));
    }
    
    let mut best_lag = 0;
    let mut best_correlation = 0.0f32;
    
    // Compute energy of the signal
    let energy: f32 = frame.iter().map(|&x| x * x).sum();
    if energy < 1e-10 {
        return Ok((0.0, 0.0));
    }
    
    // Find the lag with the highest autocorrelation
    for lag in min_lag..max_lag {
        if lag >= frame.len() {
            break;
        }
        
        let mut correlation = 0.0f32;
        let mut norm1 = 0.0f32;
        let mut norm2 = 0.0f32;
        
        for i in 0..(frame.len() - lag) {
            let x1 = frame[i];
            let x2 = frame[i + lag];
            correlation += x1 * x2;
            norm1 += x1 * x1;
            norm2 += x2 * x2;
        }
        
        // Normalize correlation
        let norm = (norm1 * norm2).sqrt();
        if norm > 1e-10 {
            correlation /= norm;
        } else {
            correlation = 0.0;
        }
        
        if correlation > best_correlation {
            best_correlation = correlation;
            best_lag = lag;
        }
    }
    
    if best_correlation > 0.3 && best_lag > 0 {
        // Minimum threshold for detection
        let frequency = sample_rate as f64 / best_lag as f64;
        Ok((frequency, best_correlation))
    } else {
        Ok((0.0, 0.0))
    }
}

/// Calculate RMS amplitude of a frame
fn calculate_rms_amplitude(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    
    let sum_squares: f32 = frame.iter().map(|&x| x * x).sum();
    (sum_squares / frame.len() as f32).sqrt()
}

/// Create a default audio configuration
#[wasm_bindgen]
pub fn create_default_audio_config() -> WasmAudioConfig {
    WasmAudioConfig::new(None, None, None, None, None, None)
}

/// Get library version information
#[wasm_bindgen]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
