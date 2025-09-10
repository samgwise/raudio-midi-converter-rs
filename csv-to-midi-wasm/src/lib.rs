use csv_to_midi_core::{
    convert_csv_string_to_midi, 
    ConversionConfig, 
    AudioEvent,
};

use csv_to_midi_core::{
    audio::{AudioAnalysisConfig, CCMappingConfig, ExtendedAudioEvent},
    midi::convert_extended_audio_to_midi,
    midi::generate_midi_file_with_cc,
    postprocess::{PostProcessingConfig, post_process_midi_with_stats},
};
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

/// Helper function to convert WASM configs to core AudioAnalysisConfig
fn create_audio_analysis_config(audio_config: &WasmAudioConfig, cc_mapping: &WasmCCMapping) -> AudioAnalysisConfig {
    AudioAnalysisConfig {
        target_sample_rate: audio_config.sample_rate as u32,
        frame_size: audio_config.frame_size,
        hop_size: audio_config.hop_size,
        fmin: audio_config.min_frequency as f64,
        fmax: audio_config.max_frequency as f64,
        threshold: audio_config.threshold as f64,
        cc_mapping: CCMappingConfig {
            pitch_contour_cc: cc_mapping.pitch_contour_cc,
            amplitude_cc: cc_mapping.amplitude_cc,
            spectral_centroid_cc: cc_mapping.spectral_centroid_cc,
            harmonicity_cc: cc_mapping.harmonicity_cc,
            spectral_rolloff_cc: cc_mapping.spectral_rolloff_cc,
            zero_crossing_rate_cc: cc_mapping.zero_crossing_rate_cc,
        },
        enable_spectral_analysis: audio_config.enable_spectral_analysis,
        enable_harmonicity_analysis: audio_config.enable_harmonicity_analysis,
        enable_zero_crossing_analysis: audio_config.enable_zero_crossing_analysis,
        enable_peak_normalization: audio_config.enable_peak_normalization,
        normalization_target: audio_config.normalization_target,
    }
}

/// Convert CSV string to MIDI byte array with optional post-processing
/// 
/// # Arguments
/// * `csv_data` - CSV string containing audio analysis data
/// * `config` - Configuration object for conversion options
/// * `postprocess_config` - Post-processing configuration (optional)
/// 
/// # Returns
/// * Uint8Array containing MIDI file data, or throws error if conversion fails
#[wasm_bindgen]
pub fn convert_csv_to_midi_wasm(
    csv_data: &str,
    config: &WasmConversionConfig,
    postprocess_config: Option<WasmPostProcessingConfig>,
) -> Result<js_sys::Uint8Array, JsValue> {
    console_log!("Starting CSV to MIDI conversion...");
    console_log!("CSV data length: {} bytes", csv_data.len());
    
    let core_config: ConversionConfig = config.clone().into();
    
    // Get MIDI event collection
    let mut collection = csv_to_midi_core::convert_csv_string_to_midi_events(csv_data, &core_config)
        .map_err(|e| JsValue::from_str(&format!("Conversion error: {}", e)))?;
    
    console_log!("Generated {} note events and {} CC events", 
                 collection.note_events.len(),
                 collection.cc_events.len());
    
    // Apply post-processing if configured
    if let Some(pp_config) = postprocess_config {
        let pp_config_core = create_postprocessing_config(&pp_config);
        let (processed_collection, stats) = post_process_midi_with_stats(
            collection, 
            &pp_config_core, 
            core_config.ticks_per_quarter
        ).map_err(|e| JsValue::from_str(&format!("Post-processing error: {}", e)))?;
        
        collection = processed_collection;
        
        console_log!("Post-processing applied:");
        console_log!("  Notes: {} -> {} ({} removed)", 
                     stats.original_note_count, 
                     stats.final_note_count, 
                     stats.original_note_count.saturating_sub(stats.final_note_count));
        console_log!("  CC Events: {} -> {} ({} simplified)", 
                     stats.original_cc_count, 
                     stats.final_cc_count, 
                     stats.cc_events_simplified);
    }
    
    // Generate MIDI file
    let midi_data = csv_to_midi_core::midi::generate_midi_file_with_cc(collection, &core_config)
        .map_err(|e| JsValue::from_str(&format!("MIDI file generation error: {}", e)))?;
    
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
    enable_spectral_analysis: bool,
    enable_harmonicity_analysis: bool,
    enable_zero_crossing_analysis: bool,
    enable_peak_normalization: bool,
    normalization_target: f32,
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
        enable_spectral_analysis: Option<bool>,
        enable_harmonicity_analysis: Option<bool>,
        enable_zero_crossing_analysis: Option<bool>,
        enable_peak_normalization: Option<bool>,
        normalization_target: Option<f32>,
    ) -> WasmAudioConfig {
        WasmAudioConfig {
            sample_rate: sample_rate.unwrap_or(44100.0),
            frame_size: frame_size.unwrap_or(2048),
            hop_size: hop_size.unwrap_or(512),
            min_frequency: min_frequency.unwrap_or(65.0),
            max_frequency: max_frequency.unwrap_or(2093.0),
            threshold: threshold.unwrap_or(0.1),
            enable_spectral_analysis: enable_spectral_analysis.unwrap_or(true),
            enable_harmonicity_analysis: enable_harmonicity_analysis.unwrap_or(true),
            enable_zero_crossing_analysis: enable_zero_crossing_analysis.unwrap_or(true),
            enable_peak_normalization: enable_peak_normalization.unwrap_or(true),
            normalization_target: normalization_target.unwrap_or(0.95),
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
    
    #[wasm_bindgen(getter)]
    pub fn enable_spectral_analysis(&self) -> bool {
        self.enable_spectral_analysis
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_enable_spectral_analysis(&mut self, value: bool) {
        self.enable_spectral_analysis = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn enable_harmonicity_analysis(&self) -> bool {
        self.enable_harmonicity_analysis
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_enable_harmonicity_analysis(&mut self, value: bool) {
        self.enable_harmonicity_analysis = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn enable_zero_crossing_analysis(&self) -> bool {
        self.enable_zero_crossing_analysis
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_enable_zero_crossing_analysis(&mut self, value: bool) {
        self.enable_zero_crossing_analysis = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn enable_peak_normalization(&self) -> bool {
        self.enable_peak_normalization
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_enable_peak_normalization(&mut self, value: bool) {
        self.enable_peak_normalization = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn normalization_target(&self) -> f32 {
        self.normalization_target
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_normalization_target(&mut self, value: f32) {
        self.normalization_target = value;
    }
}

/// Process audio samples and convert to MIDI with configurable CC mapping and post-processing
/// 
/// # Arguments
/// * `samples` - Float32Array of audio samples
/// * `audio_config` - Audio analysis configuration
/// * `cc_mapping` - CC mapping configuration
/// * `midi_config` - MIDI conversion configuration
/// * `postprocess_config` - Post-processing configuration (optional)
/// 
/// # Returns
/// * Uint8Array containing MIDI file data
#[wasm_bindgen]
pub fn process_audio_to_midi(
    samples: &[f32],
    audio_config: &WasmAudioConfig,
    cc_mapping: &WasmCCMapping,
    midi_config: &WasmConversionConfig,
    postprocess_config: Option<WasmPostProcessingConfig>,
) -> Result<js_sys::Uint8Array, JsValue> {
    console_log!("Starting enhanced audio processing...");
    console_log!("Audio samples: {} at {}Hz", samples.len(), audio_config.sample_rate);
    
    // Perform enhanced audio analysis with configurable CC mapping
    let extended_events = analyze_audio_samples_enhanced(
        samples,
        audio_config.sample_rate,
        audio_config.frame_size,
        audio_config.hop_size,
        audio_config.min_frequency,
        audio_config.max_frequency,
        audio_config.threshold,
        audio_config.enable_spectral_analysis,
        audio_config.enable_harmonicity_analysis,
        audio_config.enable_zero_crossing_analysis,
        audio_config.enable_peak_normalization,
        audio_config.normalization_target,
    )?;
    
    console_log!("Generated {} extended events", extended_events.len());
    
    // Convert WASM CC mapping to core CC mapping
    let cc_mapping_config = CCMappingConfig {
        pitch_contour_cc: cc_mapping.pitch_contour_cc,
        amplitude_cc: cc_mapping.amplitude_cc,
        spectral_centroid_cc: cc_mapping.spectral_centroid_cc,
        harmonicity_cc: cc_mapping.harmonicity_cc,
        spectral_rolloff_cc: cc_mapping.spectral_rolloff_cc,
        zero_crossing_rate_cc: cc_mapping.zero_crossing_rate_cc,
    };
    
    let core_config: ConversionConfig = midi_config.clone().into();
    
    // Convert to MIDI with configurable CC mapping
    let mut midi_event_collection = convert_extended_audio_to_midi(
        &extended_events, 
        &cc_mapping_config, 
        &core_config
    ).map_err(|e| JsValue::from_str(&format!("MIDI event conversion error: {}", e)))?;
    
    console_log!("Generated {} note events and {} CC events", 
                 midi_event_collection.note_events.len(),
                 midi_event_collection.cc_events.len());
    
    // Apply post-processing if configured
    if let Some(pp_config) = postprocess_config {
        let pp_config_core = create_postprocessing_config(&pp_config);
        let (processed_collection, stats) = post_process_midi_with_stats(
            midi_event_collection, 
            &pp_config_core, 
            core_config.ticks_per_quarter
        ).map_err(|e| JsValue::from_str(&format!("Post-processing error: {}", e)))?;
        
        midi_event_collection = processed_collection;
        
        console_log!("Post-processing applied:");
        console_log!("  Notes: {} -> {} ({} removed)", 
                     stats.original_note_count, 
                     stats.final_note_count, 
                     stats.original_note_count.saturating_sub(stats.final_note_count));
        console_log!("  CC Events: {} -> {} ({} simplified)", 
                     stats.original_cc_count, 
                     stats.final_cc_count, 
                     stats.cc_events_simplified);
    }
    
    // Generate MIDI file with CC data
    let midi_data = generate_midi_file_with_cc(midi_event_collection, &core_config)
        .map_err(|e| JsValue::from_str(&format!("MIDI file generation error: {}", e)))?;
    
    console_log!("Enhanced audio to MIDI conversion successful! Generated {} bytes", midi_data.len());
    
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

/// Enhanced audio analysis with configurable features
fn analyze_audio_samples_enhanced(
    samples: &[f32],
    sample_rate: f32,
    frame_size: usize,
    hop_size: usize,
    fmin: f32,
    fmax: f32,
    threshold: f32,
    enable_spectral_analysis: bool,
    enable_harmonicity_analysis: bool,
    enable_zero_crossing_analysis: bool,
    enable_peak_normalization: bool,
    normalization_target: f32,
) -> Result<Vec<ExtendedAudioEvent>, JsValue> {
    let mut working_samples = samples.to_vec();
    
    // Apply peak normalization if enabled
    if enable_peak_normalization {
        apply_peak_normalization_wasm(&mut working_samples, normalization_target);
    }
    
    let mut extended_events = Vec::new();
    
    // Process audio in overlapping frames
    for frame_start in (0..working_samples.len()).step_by(hop_size) {
        if frame_start + frame_size > working_samples.len() {
            break;
        }
        
        let frame = &working_samples[frame_start..frame_start + frame_size];
        let time = frame_start as f64 / sample_rate as f64;
        
        // Basic pitch detection using autocorrelation
        let (frequency, confidence) = estimate_pitch_autocorr(frame, sample_rate, fmin, fmax)?;
        let voiced = frequency > 0.0 && confidence > threshold;
        
        // Enhanced spectral analysis
        let (spectral_centroid, spectral_rolloff) = if enable_spectral_analysis {
            let centroid = calculate_spectral_centroid_wasm(frame, sample_rate as f64);
            let rolloff = calculate_spectral_rolloff_wasm(frame, sample_rate as f64, 0.85);
            (Some(centroid), Some(rolloff))
        } else {
            (None, None)
        };
        
        let zero_crossing_rate = if enable_zero_crossing_analysis {
            Some(calculate_zero_crossing_rate_wasm(frame))
        } else {
            None
        };
        
        // Calculate harmonicity (pitch clarity) if enabled
        let harmonicity = if enable_harmonicity_analysis && voiced {
            Some(calculate_harmonicity_wasm(frame, frequency, sample_rate as f64))
        } else {
            None
        };
        
        // Create pitch event
        let pitch_event = csv_to_midi_core::audio::PitchEvent {
            time,
            frequency: if voiced { frequency } else { 0.0 },
            confidence: confidence as f64,
            voiced,
            spectral_centroid,
            harmonicity,
            spectral_rolloff,
            zero_crossing_rate,
        };
        
        let extended_event: ExtendedAudioEvent = pitch_event.into();
        extended_events.push(extended_event);
    }
    
    Ok(extended_events)
}

/// Apply peak normalization to audio samples (WASM version)
fn apply_peak_normalization_wasm(samples: &mut [f32], target_level: f32) {
    if samples.is_empty() || target_level <= 0.0 || target_level > 1.0 {
        return;
    }
    
    let peak = samples.iter()
        .map(|&sample| sample.abs())
        .fold(0.0f32, |max_val, sample| max_val.max(sample));
    
    if peak > 1e-6 {
        let gain = target_level / peak;
        for sample in samples.iter_mut() {
            *sample *= gain;
        }
    }
}

/// Calculate spectral centroid (brightness measure) using simple DFT
fn calculate_spectral_centroid_wasm(frame: &[f32], sample_rate: f64) -> f64 {
    if frame.len() < 4 {
        return 0.0;
    }
    
    let n = frame.len().min(512); // Limit for performance
    let mut weighted_sum = 0.0;
    let mut magnitude_sum = 0.0;
    let bin_width = sample_rate / n as f64;
    
    // Simple DFT for first half of spectrum
    for k in 1..n/2 {
        let mut real = 0.0;
        let mut imag = 0.0;
        
        for (i, &sample) in frame.iter().enumerate().take(n) {
            let angle = -2.0 * std::f32::consts::PI * k as f32 * i as f32 / n as f32;
            real += sample * angle.cos();
            imag += sample * angle.sin();
        }
        
        let magnitude = (real * real + imag * imag).sqrt() as f64;
        let frequency = k as f64 * bin_width;
        
        weighted_sum += frequency * magnitude;
        magnitude_sum += magnitude;
    }
    
    if magnitude_sum > 1e-10 {
        weighted_sum / magnitude_sum
    } else {
        0.0
    }
}

/// Calculate spectral rolloff (frequency below which a percentage of energy is contained)
fn calculate_spectral_rolloff_wasm(frame: &[f32], sample_rate: f64, rolloff_percentage: f64) -> f64 {
    if frame.len() < 4 {
        return 0.0;
    }
    
    let n = frame.len().min(512); // Limit for performance
    let mut magnitudes = Vec::new();
    let bin_width = sample_rate / n as f64;
    
    // Simple DFT for first half of spectrum
    for k in 1..n/2 {
        let mut real = 0.0;
        let mut imag = 0.0;
        
        for (i, &sample) in frame.iter().enumerate().take(n) {
            let angle = -2.0 * std::f32::consts::PI * k as f32 * i as f32 / n as f32;
            real += sample * angle.cos();
            imag += sample * angle.sin();
        }
        
        let magnitude = (real * real + imag * imag).sqrt() as f64;
        magnitudes.push(magnitude);
    }
    
    let total_energy: f64 = magnitudes.iter().map(|&m| m * m).sum();
    let target_energy = total_energy * rolloff_percentage;
    
    let mut cumulative_energy = 0.0;
    for (i, &magnitude) in magnitudes.iter().enumerate() {
        cumulative_energy += magnitude * magnitude;
        if cumulative_energy >= target_energy {
            return (i + 1) as f64 * bin_width;
        }
    }
    
    sample_rate / 2.0
}

/// Calculate zero crossing rate (measure of noisiness)
fn calculate_zero_crossing_rate_wasm(frame: &[f32]) -> f64 {
    if frame.len() < 2 {
        return 0.0;
    }
    
    let mut crossings = 0;
    for i in 1..frame.len() {
        if (frame[i] >= 0.0) != (frame[i - 1] >= 0.0) {
            crossings += 1;
        }
    }
    
    crossings as f64 / (frame.len() - 1) as f64
}

/// Calculate harmonicity (pitch clarity measure) based on harmonic-to-noise ratio
fn calculate_harmonicity_wasm(frame: &[f32], fundamental_freq: f64, sample_rate: f64) -> f64 {
    if fundamental_freq <= 0.0 || frame.len() < 64 {
        return 0.0;
    }
    
    let period_samples = (sample_rate / fundamental_freq) as usize;
    if period_samples >= frame.len() / 2 {
        return 0.0;
    }
    
    // Calculate normalized autocorrelation at the fundamental period
    let mut correlation = 0.0;
    let mut energy1 = 0.0;
    let mut energy2 = 0.0;
    
    let len = frame.len() - period_samples;
    for i in 0..len {
        let x1 = frame[i] as f64;
        let x2 = frame[i + period_samples] as f64;
        correlation += x1 * x2;
        energy1 += x1 * x1;
        energy2 += x2 * x2;
    }
    
    let norm = (energy1 * energy2).sqrt();
    if norm > 1e-10 {
        (correlation / norm).max(0.0).min(1.0)
    } else {
        0.0
    }
}

/// WASM-compatible CC mapping configuration
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct WasmCCMapping {
    pitch_contour_cc: Option<u8>,
    amplitude_cc: Option<u8>,
    spectral_centroid_cc: Option<u8>,
    harmonicity_cc: Option<u8>,
    spectral_rolloff_cc: Option<u8>,
    zero_crossing_rate_cc: Option<u8>,
}

#[wasm_bindgen]
impl WasmCCMapping {
    #[wasm_bindgen(constructor)]
    pub fn new(
        pitch_contour_cc: Option<u8>,
        amplitude_cc: Option<u8>,
        spectral_centroid_cc: Option<u8>,
        harmonicity_cc: Option<u8>,
        spectral_rolloff_cc: Option<u8>,
        zero_crossing_rate_cc: Option<u8>,
    ) -> WasmCCMapping {
        WasmCCMapping {
            pitch_contour_cc,
            amplitude_cc,
            spectral_centroid_cc,
            harmonicity_cc,
            spectral_rolloff_cc,
            zero_crossing_rate_cc,
        }
    }
    
    // Getters and setters for each CC assignment
    #[wasm_bindgen(getter)]
    pub fn pitch_contour_cc(&self) -> Option<u8> {
        self.pitch_contour_cc
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_pitch_contour_cc(&mut self, cc: Option<u8>) {
        self.pitch_contour_cc = cc;
    }
    
    #[wasm_bindgen(getter)]
    pub fn amplitude_cc(&self) -> Option<u8> {
        self.amplitude_cc
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_amplitude_cc(&mut self, cc: Option<u8>) {
        self.amplitude_cc = cc;
    }
    
    #[wasm_bindgen(getter)]
    pub fn spectral_centroid_cc(&self) -> Option<u8> {
        self.spectral_centroid_cc
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_spectral_centroid_cc(&mut self, cc: Option<u8>) {
        self.spectral_centroid_cc = cc;
    }
    
    #[wasm_bindgen(getter)]
    pub fn harmonicity_cc(&self) -> Option<u8> {
        self.harmonicity_cc
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_harmonicity_cc(&mut self, cc: Option<u8>) {
        self.harmonicity_cc = cc;
    }
    
    #[wasm_bindgen(getter)]
    pub fn spectral_rolloff_cc(&self) -> Option<u8> {
        self.spectral_rolloff_cc
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_spectral_rolloff_cc(&mut self, cc: Option<u8>) {
        self.spectral_rolloff_cc = cc;
    }
    
    #[wasm_bindgen(getter)]
    pub fn zero_crossing_rate_cc(&self) -> Option<u8> {
        self.zero_crossing_rate_cc
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_zero_crossing_rate_cc(&mut self, cc: Option<u8>) {
        self.zero_crossing_rate_cc = cc;
    }
}

/// WASM-compatible post-processing configuration
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct WasmPostProcessingConfig {
    enable_pitch_filtering: bool,
    min_midi_note: u8,
    max_midi_note: u8,
    
    enable_velocity_expansion: bool,
    velocity_threshold: u8,
    velocity_expansion_factor: f32,
    max_expanded_velocity: u8,
    
    enable_note_joining: bool,
    min_note_duration: u32,
    max_join_gap: u32,
    remove_short_notes_threshold: u32,
    
    enable_duplicate_removal: bool,
    duplicate_time_window: u32,
    
    enable_cc_simplification: bool,
    cc_min_change_threshold: u8,
    cc_max_events_per_second: f32,
}

#[wasm_bindgen]
impl WasmPostProcessingConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(
        enable_pitch_filtering: Option<bool>,
        min_midi_note: Option<u8>,
        max_midi_note: Option<u8>,
        
        enable_velocity_expansion: Option<bool>,
        velocity_threshold: Option<u8>,
        velocity_expansion_factor: Option<f32>,
        max_expanded_velocity: Option<u8>,
        
        enable_note_joining: Option<bool>,
        min_note_duration: Option<u32>,
        max_join_gap: Option<u32>,
        remove_short_notes_threshold: Option<u32>,
        
        enable_duplicate_removal: Option<bool>,
        duplicate_time_window: Option<u32>,
        
        enable_cc_simplification: Option<bool>,
        cc_min_change_threshold: Option<u8>,
        cc_max_events_per_second: Option<f32>,
    ) -> WasmPostProcessingConfig {
        WasmPostProcessingConfig {
            enable_pitch_filtering: enable_pitch_filtering.unwrap_or(false),
            min_midi_note: min_midi_note.unwrap_or(21),
            max_midi_note: max_midi_note.unwrap_or(108),
            
            enable_velocity_expansion: enable_velocity_expansion.unwrap_or(false),
            velocity_threshold: velocity_threshold.unwrap_or(40),
            velocity_expansion_factor: velocity_expansion_factor.unwrap_or(1.5),
            max_expanded_velocity: max_expanded_velocity.unwrap_or(100),
            
            enable_note_joining: enable_note_joining.unwrap_or(true),
            min_note_duration: min_note_duration.unwrap_or(50),
            max_join_gap: max_join_gap.unwrap_or(24),
            remove_short_notes_threshold: remove_short_notes_threshold.unwrap_or(12),
            
            enable_duplicate_removal: enable_duplicate_removal.unwrap_or(true),
            duplicate_time_window: duplicate_time_window.unwrap_or(12),
            
            enable_cc_simplification: enable_cc_simplification.unwrap_or(true),
            cc_min_change_threshold: cc_min_change_threshold.unwrap_or(2),
            cc_max_events_per_second: cc_max_events_per_second.unwrap_or(20.0),
        }
    }
    
    // Getters and setters for each field
    #[wasm_bindgen(getter)]
    pub fn enable_pitch_filtering(&self) -> bool {
        self.enable_pitch_filtering
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_enable_pitch_filtering(&mut self, value: bool) {
        self.enable_pitch_filtering = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn min_midi_note(&self) -> u8 {
        self.min_midi_note
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_min_midi_note(&mut self, value: u8) {
        self.min_midi_note = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn max_midi_note(&self) -> u8 {
        self.max_midi_note
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_max_midi_note(&mut self, value: u8) {
        self.max_midi_note = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn enable_velocity_expansion(&self) -> bool {
        self.enable_velocity_expansion
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_enable_velocity_expansion(&mut self, value: bool) {
        self.enable_velocity_expansion = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn velocity_threshold(&self) -> u8 {
        self.velocity_threshold
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_velocity_threshold(&mut self, value: u8) {
        self.velocity_threshold = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn velocity_expansion_factor(&self) -> f32 {
        self.velocity_expansion_factor
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_velocity_expansion_factor(&mut self, value: f32) {
        self.velocity_expansion_factor = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn enable_note_joining(&self) -> bool {
        self.enable_note_joining
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_enable_note_joining(&mut self, value: bool) {
        self.enable_note_joining = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn max_join_gap(&self) -> u32 {
        self.max_join_gap
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_max_join_gap(&mut self, value: u32) {
        self.max_join_gap = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn enable_cc_simplification(&self) -> bool {
        self.enable_cc_simplification
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_enable_cc_simplification(&mut self, value: bool) {
        self.enable_cc_simplification = value;
    }
}

/// Helper function to convert WASM post-processing config to core config
fn create_postprocessing_config(config: &WasmPostProcessingConfig) -> PostProcessingConfig {
    PostProcessingConfig {
        enable_pitch_filtering: config.enable_pitch_filtering,
        min_midi_note: config.min_midi_note,
        max_midi_note: config.max_midi_note,
        
        enable_velocity_expansion: config.enable_velocity_expansion,
        velocity_threshold: config.velocity_threshold,
        velocity_expansion_factor: config.velocity_expansion_factor,
        max_expanded_velocity: config.max_expanded_velocity,
        
        enable_note_joining: config.enable_note_joining,
        min_note_duration: config.min_note_duration,
        max_join_gap: config.max_join_gap,
        remove_short_notes_threshold: config.remove_short_notes_threshold,
        
        enable_duplicate_removal: config.enable_duplicate_removal,
        duplicate_time_window: config.duplicate_time_window,
        
        enable_cc_simplification: config.enable_cc_simplification,
        cc_min_change_threshold: config.cc_min_change_threshold,
        cc_max_events_per_second: config.cc_max_events_per_second,
    }
}

// Note: Advanced CC mapping is not available in the WASM version
// This is maintained for API compatibility but doesn't map to core functionality

/// Create a default CC mapping configuration
#[wasm_bindgen]
pub fn create_default_cc_mapping() -> WasmCCMapping {
    WasmCCMapping::new(
        Some(100), // Pitch contour
        Some(101), // Amplitude  
        Some(102), // Spectral centroid
        Some(103), // Harmonicity
        Some(104), // Spectral rolloff
        Some(105), // Zero crossing rate
    )
}

/// Create a default audio configuration
#[wasm_bindgen]
pub fn create_default_audio_config() -> WasmAudioConfig {
    WasmAudioConfig::new(None, None, None, None, None, None, None, None, None, None, None)
}

/// Create a default post-processing configuration
#[wasm_bindgen]
pub fn create_default_postprocessing_config() -> WasmPostProcessingConfig {
    WasmPostProcessingConfig::new(
        None, None, None,  // pitch filtering disabled by default
        None, None, None, None,  // velocity expansion disabled by default 
        None, None, None, None,  // note joining enabled by default
        None, None,  // duplicate removal enabled by default
        None, None, None  // CC simplification enabled by default
    )
}

/// Get library version information
#[wasm_bindgen]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
