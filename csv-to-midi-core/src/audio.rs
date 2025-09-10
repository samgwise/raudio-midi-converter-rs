//! Audio analysis module using PYIN pitch detection
//!
//! This module provides functionality to analyze audio files (WAV, FLAC, etc.)
//! using the PYIN algorithm for pitch detection and convert the results to MIDI.

use crate::{ConversionConfig, ConversionError, Result};
use crate::parser::AudioEvent;
use std::path::Path;

#[cfg(feature = "audio")]
use {
    hound::{WavReader, SampleFormat},
    rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction},
    rustfft::{FftPlanner, num_complex::Complex},
};

/// MIDI Controller Change mapping configuration
#[derive(Debug, Clone)]
pub struct CCMappingConfig {
    /// CC number for pitch contour (fine pitch variations)
    pub pitch_contour_cc: Option<u8>,
    /// CC number for amplitude/volume envelope
    pub amplitude_cc: Option<u8>,
    /// CC number for spectral centroid (brightness)
    pub spectral_centroid_cc: Option<u8>,
    /// CC number for harmonicity (pitch clarity)
    pub harmonicity_cc: Option<u8>,
    /// CC number for spectral rolloff (frequency distribution)
    pub spectral_rolloff_cc: Option<u8>,
    /// CC number for zero crossing rate (noisiness)
    pub zero_crossing_rate_cc: Option<u8>,
}

impl Default for CCMappingConfig {
    fn default() -> Self {
        Self {
            pitch_contour_cc: Some(100),     // CC 100: Pitch contour
            amplitude_cc: Some(101),         // CC 101: Amplitude
            spectral_centroid_cc: Some(102), // CC 102: Spectral centroid (brightness)
            harmonicity_cc: Some(103),      // CC 103: Harmonicity (pitch clarity)
            spectral_rolloff_cc: Some(104), // CC 104: Spectral rolloff
            zero_crossing_rate_cc: Some(105), // CC 105: Zero crossing rate
        }
    }
}

/// Configuration specific to audio analysis
#[derive(Debug, Clone)]
pub struct AudioAnalysisConfig {
    /// Sample rate for analysis (works best at 44.1kHz)
    pub target_sample_rate: u32,
    /// Frame size for analysis in samples
    pub frame_size: usize,
    /// Hop size for analysis in samples (frame overlap)
    pub hop_size: usize,
    /// Minimum frequency to detect in Hz
    pub fmin: f64,
    /// Maximum frequency to detect in Hz
    pub fmax: f64,
    /// Threshold for voiced/unvoiced detection (0.0-1.0)
    pub threshold: f64,
    /// CC mapping configuration
    pub cc_mapping: CCMappingConfig,
    /// Enable advanced spectral analysis (spectral centroid, rolloff, etc.)
    pub enable_spectral_analysis: bool,
    /// Enable harmonicity analysis (pitch clarity measure)
    pub enable_harmonicity_analysis: bool,
    /// Enable zero crossing rate analysis (noisiness measure)
    pub enable_zero_crossing_analysis: bool,
    /// Enable peak normalization before analysis (default: true)
    pub enable_peak_normalization: bool,
    /// Peak normalization target level (0.0-1.0, default: 0.95)
    pub normalization_target: f32,
    /// Use pYIN algorithm for pitch detection instead of autocorrelation (default: true)
    #[cfg(feature = "pyin")]
    pub use_pyin_pitch_detection: bool,
    /// pYIN resolution parameter (None for default: 0.1)
    #[cfg(feature = "pyin")]
    pub pyin_resolution: Option<f64>,
}

impl Default for AudioAnalysisConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 44100,
            frame_size: 2048,
            hop_size: 512,  // 25% overlap
            fmin: 65.0,     // C2
            fmax: 2093.0,   // C7
            threshold: 0.1, // Default threshold
            cc_mapping: CCMappingConfig::default(),
            enable_spectral_analysis: true,
            enable_harmonicity_analysis: true,
            enable_zero_crossing_analysis: true,
            enable_peak_normalization: true,
            normalization_target: 0.95, // Leave some headroom
            #[cfg(feature = "pyin")]
            use_pyin_pitch_detection: true,
            #[cfg(feature = "pyin")]
            pyin_resolution: Some(0.05), // Higher resolution for better accuracy
        }
    }
}

/// Represents the result of audio pitch analysis
#[derive(Debug, Clone)]
pub struct PitchAnalysisResult {
    /// Detected pitch candidates with timestamps and probabilities
    pub pitch_events: Vec<PitchEvent>,
    /// Original sample rate of the audio
    pub sample_rate: u32,
    /// Total duration of the analyzed audio in seconds
    pub duration: f64,
}

/// A single pitch detection event with extended analysis data
#[derive(Debug, Clone)]
pub struct PitchEvent {
    /// Time offset in seconds
    pub time: f64,
    /// Detected frequency in Hz (0.0 if unvoiced)
    pub frequency: f64,
    /// Confidence/probability of the detection (0.0-1.0)
    pub confidence: f64,
    /// Whether this frame is considered voiced
    pub voiced: bool,
    /// Spectral centroid (brightness measure in Hz)
    pub spectral_centroid: Option<f64>,
    /// Harmonicity (pitch clarity measure 0.0-1.0)
    pub harmonicity: Option<f64>,
    /// Spectral rolloff frequency (Hz)
    pub spectral_rolloff: Option<f64>,
    /// Zero crossing rate (measure of noisiness)
    pub zero_crossing_rate: Option<f64>,
}

impl From<PitchEvent> for AudioEvent {
    fn from(pitch_event: PitchEvent) -> Self {
        AudioEvent {
            line_number: 0, // Will be set by the caller
            timestamp: pitch_event.time,
            frequency: if pitch_event.voiced { pitch_event.frequency } else { 0.0 },
            amplitude: pitch_event.confidence,
            channel: Some(1), // Single channel for audio analysis
        }
    }
}

/// Extended audio event with analysis data for CC generation
#[derive(Debug, Clone)]
pub struct ExtendedAudioEvent {
    pub base_event: AudioEvent,
    pub pitch_event: PitchEvent,
}

impl From<PitchEvent> for ExtendedAudioEvent {
    fn from(pitch_event: PitchEvent) -> Self {
        let base_event = AudioEvent {
            line_number: 0,
            timestamp: pitch_event.time,
            frequency: if pitch_event.voiced { pitch_event.frequency } else { 0.0 },
            amplitude: pitch_event.confidence,
            channel: Some(1),
        };
        
        ExtendedAudioEvent {
            base_event,
            pitch_event,
        }
    }
}

#[cfg(feature = "audio")]
/// Analyze audio file using PYIN algorithm
pub fn analyze_audio_file<P: AsRef<Path>>(
    file_path: P,
    audio_config: &AudioAnalysisConfig,
) -> Result<PitchAnalysisResult> {
    let file_path = file_path.as_ref();
    
    // Read the audio file
    let mut reader = WavReader::open(file_path)
        .map_err(|e| ConversionError::IoError(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to open audio file: {}", e)
        )))?;
    
    let spec = reader.spec();
    let original_sample_rate = spec.sample_rate;
    
    // Read samples based on the sample format
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => {
            reader.samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| ConversionError::IoError(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to read audio samples: {}", e)
                )))?
        }
        SampleFormat::Int => {
            let int_samples: Vec<i32> = reader.samples::<i32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| ConversionError::IoError(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to read audio samples: {}", e)
                )))?;
            
            // Convert to float and normalize
            let max_val = match spec.bits_per_sample {
                16 => i16::MAX as f32,
                24 => 8388607.0, // 2^23 - 1
                32 => i32::MAX as f32,
                _ => return Err(ConversionError::InvalidFrequency(-1.0)),
            };
            
            int_samples.into_iter()
                .map(|sample| sample as f32 / max_val)
                .collect()
        }
    };
    
    // Convert to mono if stereo
    let mono_samples: Vec<f32> = if spec.channels == 1 {
        samples
    } else {
        samples.chunks(spec.channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
            .collect()
    };
    
    // Resample if necessary
    let resampled_samples = if original_sample_rate != audio_config.target_sample_rate {
        resample_audio(&mono_samples, original_sample_rate, audio_config.target_sample_rate)?
    } else {
        mono_samples
    };
    
    // Analyze with PYIN
    analyze_with_pyin(&resampled_samples, audio_config)
}

#[cfg(feature = "audio")]
/// Resample audio to target sample rate
fn resample_audio(
    samples: &[f32],
    from_rate: u32,
    to_rate: u32,
) -> Result<Vec<f32>> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }
    
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    
    let mut resampler = SincFixedIn::<f32>::new(
        to_rate as f64 / from_rate as f64,
        2.0, // Max relative ratio change
        params,
        samples.len(),
        1, // Single channel
    ).map_err(|e| ConversionError::MidiWriteError(format!("Resampling failed: {:?}", e)))?;
    
    let input = vec![samples.to_vec()]; // Single channel
    let output = resampler.process(&input, None)
        .map_err(|e| ConversionError::MidiWriteError(format!("Resampling failed: {:?}", e)))?;
    
    Ok(output[0].clone())
}

#[cfg(feature = "audio")]
/// Analyze audio samples with enhanced spectral and temporal analysis
fn analyze_with_pyin(
    samples: &[f32],
    config: &AudioAnalysisConfig,
) -> Result<PitchAnalysisResult> {
    let sample_rate = config.target_sample_rate;
    let duration = samples.len() as f64 / sample_rate as f64;
    
    // Create a mutable copy of samples for potential normalization
    let mut working_samples = samples.to_vec();
    
    // Apply peak normalization if enabled
    if config.enable_peak_normalization {
        apply_peak_normalization(&mut working_samples, config.normalization_target);
    }
    
    let mut pitch_events = Vec::new();
    
    // Choose pitch detection method
    #[cfg(feature = "pyin")]
    if config.use_pyin_pitch_detection {
        // Use pYIN for advanced pitch detection
        let pyin_config = crate::pyin_pitch::PyinConfig {
            sample_rate,
            frame_length: config.frame_size,
            hop_length: Some(config.hop_size),
            fmin: config.fmin,
            fmax: config.fmax,
            resolution: config.pyin_resolution,
            ..Default::default()
        };
        
        let pyin_detector = crate::pyin_pitch::PyinPitchDetector::new(pyin_config);
        let pyin_result = pyin_detector.analyze(&working_samples)?;
        
        // Initialize FFT planner for spectral analysis
        let mut fft_planner = if config.enable_spectral_analysis {
            Some(FftPlanner::new())
        } else {
            None
        };
        
        // Convert pYIN results to PitchEvent format with additional spectral analysis
        for (i, &time) in pyin_result.timestamps.iter().enumerate() {
            let frequency = pyin_result.frequencies[i];
            let voiced = pyin_result.voiced_flags[i] && !frequency.is_nan();
            let confidence = pyin_result.voiced_probabilities[i];
            
            // Calculate frame boundaries for spectral analysis
            let frame_start = (time * sample_rate as f64) as usize;
            let frame_end = (frame_start + config.frame_size).min(working_samples.len());
            
            let (spectral_centroid, spectral_rolloff, zero_crossing_rate, harmonicity) = 
                if frame_start < working_samples.len() && frame_end > frame_start {
                    let frame = &working_samples[frame_start..frame_end];
                    
                    // Enhanced spectral analysis
                    let (centroid, rolloff) = if config.enable_spectral_analysis {
                        if let Some(ref mut planner) = fft_planner {
                            let centroid = calculate_spectral_centroid(frame, sample_rate as f64, planner);
                            let rolloff = calculate_spectral_rolloff(frame, sample_rate as f64, planner, 0.85);
                            (Some(centroid), Some(rolloff))
                        } else {
                            (None, None)
                        }
                    } else {
                        (None, None)
                    };
                    
                    let zcr = if config.enable_zero_crossing_analysis {
                        Some(calculate_zero_crossing_rate(frame))
                    } else {
                        None
                    };
                    
                    // Calculate harmonicity if enabled
                    let harm = if config.enable_harmonicity_analysis && voiced && frequency > 0.0 {
                        Some(calculate_harmonicity(frame, frequency, sample_rate as f64))
                    } else {
                        None
                    };
                    
                    (centroid, rolloff, zcr, harm)
                } else {
                    (None, None, None, None)
                };
            
            pitch_events.push(PitchEvent {
                time,
                frequency: if voiced { frequency } else { 0.0 },
                confidence,
                voiced,
                spectral_centroid,
                harmonicity,
                spectral_rolloff,
                zero_crossing_rate,
            });
        }
    }
    #[cfg(not(feature = "pyin"))]
    {
        // Use frame-based analysis with autocorrelation when pYIN is not available
        let frame_size = config.frame_size;
        let hop_size = config.hop_size;
        
        // Initialize FFT planner for spectral analysis
        let mut fft_planner = if config.enable_spectral_analysis {
            Some(FftPlanner::new())
        } else {
            None
        };
        
        // Process audio in overlapping frames
        for frame_start in (0..working_samples.len()).step_by(hop_size) {
            if frame_start + frame_size > working_samples.len() {
                break;
            }
            
            let frame = &working_samples[frame_start..frame_start + frame_size];
            let time = frame_start as f64 / sample_rate as f64;
            
            // Basic pitch detection using autocorrelation
            let (frequency, confidence) = estimate_pitch_autocorr(frame, sample_rate, config.fmin, config.fmax);
            let voiced = frequency > 0.0 && confidence > config.threshold;
            
            // Enhanced spectral analysis using custom functions
            let (spectral_centroid, spectral_rolloff) = if config.enable_spectral_analysis {
                if let Some(ref mut planner) = fft_planner {
                    let centroid = calculate_spectral_centroid(frame, sample_rate as f64, planner);
                    let rolloff = calculate_spectral_rolloff(frame, sample_rate as f64, planner, 0.85);
                    (Some(centroid), Some(rolloff))
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            };
            
            let zero_crossing_rate = if config.enable_zero_crossing_analysis {
                Some(calculate_zero_crossing_rate(frame))
            } else {
                None
            };
            
            // Calculate harmonicity (pitch clarity) if enabled
            let harmonicity = if config.enable_harmonicity_analysis && voiced {
                Some(calculate_harmonicity(frame, frequency, sample_rate as f64))
            } else {
                None
            };
            
            pitch_events.push(PitchEvent {
                time,
                frequency: if voiced { frequency } else { 0.0 },
                confidence,
                voiced,
                spectral_centroid,
                harmonicity,
                spectral_rolloff,
                zero_crossing_rate,
            });
        }
    }
    
    Ok(PitchAnalysisResult {
        pitch_events,
        sample_rate,
        duration,
    })
}

#[cfg(feature = "audio")]
/// Calculate spectral centroid (brightness measure) using FFT
fn calculate_spectral_centroid(frame: &[f32], sample_rate: f64, fft_planner: &mut FftPlanner<f32>) -> f64 {
    if frame.len() < 4 {
        return 0.0;
    }
    
    // Prepare FFT input
    let mut buffer: Vec<Complex<f32>> = frame.iter().map(|&x| Complex::new(x, 0.0)).collect();
    
    // Ensure power of 2 length for FFT efficiency
    let fft_size = buffer.len().next_power_of_two();
    buffer.resize(fft_size, Complex::new(0.0, 0.0));
    
    // Perform FFT
    let fft = fft_planner.plan_fft_forward(fft_size);
    fft.process(&mut buffer);
    
    // Calculate spectral centroid
    let mut weighted_sum = 0.0;
    let mut magnitude_sum = 0.0;
    let bin_width = sample_rate / fft_size as f64;
    
    for (i, complex) in buffer.iter().take(fft_size / 2).enumerate() {
        let magnitude = complex.norm() as f64;
        let frequency = i as f64 * bin_width;
        
        weighted_sum += frequency * magnitude;
        magnitude_sum += magnitude;
    }
    
    if magnitude_sum > 1e-10 {
        weighted_sum / magnitude_sum
    } else {
        0.0
    }
}

#[cfg(feature = "audio")]
/// Calculate spectral rolloff (frequency below which a percentage of energy is contained)
fn calculate_spectral_rolloff(frame: &[f32], sample_rate: f64, fft_planner: &mut FftPlanner<f32>, rolloff_percentage: f64) -> f64 {
    if frame.len() < 4 {
        return 0.0;
    }
    
    // Prepare FFT input
    let mut buffer: Vec<Complex<f32>> = frame.iter().map(|&x| Complex::new(x, 0.0)).collect();
    
    // Ensure power of 2 length for FFT efficiency
    let fft_size = buffer.len().next_power_of_two();
    buffer.resize(fft_size, Complex::new(0.0, 0.0));
    
    // Perform FFT
    let fft = fft_planner.plan_fft_forward(fft_size);
    fft.process(&mut buffer);
    
    // Calculate total energy and find rolloff point
    let magnitudes: Vec<f64> = buffer.iter().take(fft_size / 2).map(|c| c.norm() as f64).collect();
    let total_energy: f64 = magnitudes.iter().map(|&m| m * m).sum();
    let target_energy = total_energy * rolloff_percentage;
    
    let mut cumulative_energy = 0.0;
    let bin_width = sample_rate / fft_size as f64;
    
    for (i, &magnitude) in magnitudes.iter().enumerate() {
        cumulative_energy += magnitude * magnitude;
        if cumulative_energy >= target_energy {
            return i as f64 * bin_width;
        }
    }
    
    // If we haven't found the rolloff point, return the Nyquist frequency
    sample_rate / 2.0
}

#[cfg(feature = "audio")]
/// Calculate zero crossing rate (measure of noisiness)
fn calculate_zero_crossing_rate(frame: &[f32]) -> f64 {
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

#[cfg(feature = "audio")]
/// Apply peak normalization to audio samples
/// 
/// This function normalizes the audio to a target peak level to ensure
/// consistent signal levels for analysis, improving pitch detection accuracy
/// and spectral analysis reliability.
fn apply_peak_normalization(samples: &mut [f32], target_level: f32) {
    if samples.is_empty() || target_level <= 0.0 || target_level > 1.0 {
        return;
    }
    
    // Find the peak absolute value in the audio
    let peak = samples.iter()
        .map(|&sample| sample.abs())
        .fold(0.0f32, |max_val, sample| max_val.max(sample));
    
    // Avoid division by zero and don't amplify if peak is already very small
    if peak > 1e-6 {
        let gain = target_level / peak;
        
        // Apply the gain to all samples
        for sample in samples.iter_mut() {
            *sample *= gain;
        }
    }
}

#[cfg(feature = "audio")]
/// Calculate harmonicity (pitch clarity measure) based on harmonic-to-noise ratio
fn calculate_harmonicity(frame: &[f32], fundamental_freq: f64, sample_rate: f64) -> f64 {
    if fundamental_freq <= 0.0 || frame.len() < 64 {
        return 0.0;
    }
    
    // Simple harmonicity measure based on autocorrelation strength
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

#[cfg(feature = "audio")]
/// Simple autocorrelation-based pitch estimation
fn estimate_pitch_autocorr(frame: &[f32], sample_rate: u32, fmin: f64, fmax: f64) -> (f64, f64) {
    if frame.len() < 2 {
        return (0.0, 0.0);
    }
    
    // Calculate autocorrelation
    let max_lag = (sample_rate as f64 / fmin).min(frame.len() as f64 / 2.0) as usize;
    let min_lag = (sample_rate as f64 / fmax).max(1.0) as usize;
    
    if min_lag >= max_lag {
        return (0.0, 0.0);
    }
    
    let mut best_lag = 0;
    let mut best_correlation = 0.0;
    
    // Compute energy of the signal
    let energy: f64 = frame.iter().map(|&x| (x as f64).powi(2)).sum();
    if energy < 1e-10 {
        return (0.0, 0.0);
    }
    
    // Find the lag with the highest autocorrelation
    for lag in min_lag..max_lag {
        if lag >= frame.len() {
            break;
        }
        
        let mut correlation = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;
        
        for i in 0..(frame.len() - lag) {
            let x1 = frame[i] as f64;
            let x2 = frame[i + lag] as f64;
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
    
    if best_correlation > 0.3 && best_lag > 0 { // Minimum threshold for detection
        let frequency = sample_rate as f64 / best_lag as f64;
        (frequency, best_correlation)
    } else {
        (0.0, 0.0)
    }
}

#[cfg(not(feature = "audio"))]
pub fn analyze_audio_file<P: AsRef<Path>>(
    _file_path: P,
    _audio_config: &AudioAnalysisConfig,
) -> Result<PitchAnalysisResult> {
    Err(ConversionError::MidiWriteError(
        "Audio analysis feature is not enabled".to_string()
    ))
}

/// Convert audio analysis results to AudioEvent format for MIDI conversion (note-level)
pub fn pitch_analysis_to_audio_events(
    analysis: &PitchAnalysisResult,
    min_note_duration: f64,
) -> Vec<AudioEvent> {
    let mut audio_events = Vec::new();
    let mut line_number = 1;
    
    // Group consecutive pitch events and filter short notes
    let mut current_pitch: Option<f64> = None;
    let mut current_start_time: Option<f64> = None;
    let mut current_confidence: f64 = 0.0;
    
    for pitch_event in &analysis.pitch_events {
        let pitch = if pitch_event.voiced { Some(pitch_event.frequency) } else { None };
        
        match (current_pitch, pitch) {
            // Continuing same pitch
            (Some(curr), Some(new)) if (curr - new).abs() < 5.0 => {
                // Update confidence to maximum
                current_confidence = current_confidence.max(pitch_event.confidence);
            }
            
            // Starting new pitch or changing pitch
            (_, Some(new_pitch)) => {
                // End current note if it exists and meets minimum duration
                if let (Some(_), Some(start_time)) = (current_pitch, current_start_time) {
                    let duration = pitch_event.time - start_time;
                    if duration >= min_note_duration {
                        audio_events.push(AudioEvent {
                            line_number,
                            timestamp: start_time,
                            frequency: current_pitch.unwrap(),
                            amplitude: current_confidence,
                            channel: Some(1),
                        });
                        line_number += 1;
                    }
                }
                
                // Start new note
                current_pitch = Some(new_pitch);
                current_start_time = Some(pitch_event.time);
                current_confidence = pitch_event.confidence;
            }
            
            // Ending pitch (going to silence)
            (Some(_), None) => {
                if let Some(start_time) = current_start_time {
                    let duration = pitch_event.time - start_time;
                    if duration >= min_note_duration {
                        audio_events.push(AudioEvent {
                            line_number,
                            timestamp: start_time,
                            frequency: current_pitch.unwrap(),
                            amplitude: current_confidence,
                            channel: Some(1),
                        });
                        line_number += 1;
                    }
                }
                
                // Add silence marker
                audio_events.push(AudioEvent {
                    line_number,
                    timestamp: pitch_event.time,
                    frequency: 0.0,
                    amplitude: 0.0,
                    channel: None,
                });
                line_number += 1;
                
                current_pitch = None;
                current_start_time = None;
            }
            
            // Staying in silence or starting in silence
            (None, None) => {
                // Continue silence
            }
        }
    }
    
    // Handle final note if exists
    if let (Some(_), Some(start_time)) = (current_pitch, current_start_time) {
        let duration = analysis.duration - start_time;
        if duration >= min_note_duration {
            audio_events.push(AudioEvent {
                line_number,
                timestamp: start_time,
                frequency: current_pitch.unwrap(),
                amplitude: current_confidence,
                channel: Some(1),
            });
        }
    }
    
    audio_events
}

/// Build extended events per analysis frame for CC generation
pub fn build_extended_events_from_analysis(analysis: &PitchAnalysisResult) -> Vec<ExtendedAudioEvent> {
    let mut events = Vec::new();
    for p in &analysis.pitch_events {
        let ee: ExtendedAudioEvent = p.clone().into();
        events.push(ee);
    }
    events
}

/// Convert audio file directly to MIDI using analysis
pub fn convert_audio_to_midi<P: AsRef<Path>>(
    file_path: P,
    audio_config: &AudioAnalysisConfig,
    midi_config: &ConversionConfig,
) -> Result<Vec<u8>> {
    // Analyze audio file
    let analysis = analyze_audio_file(file_path, audio_config)?;
    
    // Build extended events per analysis frame for rich CC data
    let extended_events = build_extended_events_from_analysis(&analysis);
    
    // Convert to MIDI with configurable CC mapping
    let collection = crate::midi::convert_extended_audio_to_midi(&extended_events, &audio_config.cc_mapping, midi_config)?;
    crate::midi::generate_midi_file_with_cc(collection, midi_config)
}

/// Convert audio file to MIDI event collection (for post-processing)
pub fn convert_audio_to_midi_events<P: AsRef<Path>>(
    file_path: P,
    audio_config: &AudioAnalysisConfig,
    midi_config: &ConversionConfig,
) -> Result<crate::midi::MidiEventCollection> {
    // Analyze audio file
    let analysis = analyze_audio_file(file_path, audio_config)?;
    
    // Build extended events per analysis frame for rich CC data
    let extended_events = build_extended_events_from_analysis(&analysis);
    
    // Convert to MIDI with configurable CC mapping
    crate::midi::convert_extended_audio_to_midi(&extended_events, &audio_config.cc_mapping, midi_config)
}

/// Convert audio file to MIDI with optional post-processing
pub fn convert_audio_to_midi_with_postprocess<P: AsRef<Path>>(
    file_path: P,
    audio_config: &AudioAnalysisConfig,
    midi_config: &ConversionConfig,
    postprocess_config: Option<&crate::postprocess::PostProcessingConfig>,
) -> Result<Vec<u8>> {
    // Get MIDI event collection
    let mut collection = convert_audio_to_midi_events(file_path, audio_config, midi_config)?;
    
    // Apply post-processing if configured
    if let Some(pp_config) = postprocess_config {
        let (processed_collection, stats) = crate::postprocess::post_process_midi_with_stats(collection, pp_config, midi_config.ticks_per_quarter)?;
        collection = processed_collection;
        
        // Log post-processing stats (optional)
        eprintln!("Post-processing stats:");
        eprintln!("  Notes: {} -> {} ({} removed)", stats.original_note_count, stats.final_note_count, stats.original_note_count.saturating_sub(stats.final_note_count));
        eprintln!("  CC Events: {} -> {} ({} simplified)", stats.original_cc_count, stats.final_cc_count, stats.cc_events_simplified);
    }
    
    // Generate MIDI file
    crate::midi::generate_midi_file_with_cc(collection, midi_config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "audio")]
    fn test_peak_normalization_basic() {
        let mut samples = vec![0.1, -0.2, 0.5, -0.8, 0.3];
        let target_level = 0.95;
        
        apply_peak_normalization(&mut samples, target_level);
        
        // The peak should now be approximately 0.95 (from original 0.8)
        let actual_peak = samples.iter().map(|&s| s.abs()).fold(0.0f32, |max, val| max.max(val));
        assert!((actual_peak - target_level).abs() < 0.001);
    }

    #[test]
    #[cfg(feature = "audio")]
    fn test_peak_normalization_zero_signal() {
        let mut samples = vec![0.0, 0.0, 0.0, 0.0];
        let original = samples.clone();
        
        apply_peak_normalization(&mut samples, 0.95);
        
        // Zero signal should remain unchanged
        assert_eq!(samples, original);
    }

    #[test]
    #[cfg(feature = "audio")]
    fn test_peak_normalization_small_signal() {
        let mut samples = vec![1e-8, -1e-8, 5e-9];
        let original = samples.clone();
        
        apply_peak_normalization(&mut samples, 0.95);
        
        // Very small signals below threshold should remain unchanged
        assert_eq!(samples, original);
    }

    #[test]
    #[cfg(feature = "audio")]
    fn test_peak_normalization_empty_array() {
        let mut samples: Vec<f32> = vec![];
        
        // Should not panic on empty array
        apply_peak_normalization(&mut samples, 0.95);
        assert!(samples.is_empty());
    }

    #[test]
    #[cfg(feature = "audio")]
    fn test_peak_normalization_invalid_target() {
        let mut samples = vec![0.1, -0.2, 0.5];
        let original = samples.clone();
        
        // Invalid target levels should leave samples unchanged
        apply_peak_normalization(&mut samples, 0.0);
        assert_eq!(samples, original);
        
        apply_peak_normalization(&mut samples, -0.5);
        assert_eq!(samples, original);
        
        apply_peak_normalization(&mut samples, 1.5);
        assert_eq!(samples, original);
    }

    #[test]
    #[cfg(feature = "audio")]
    fn test_peak_normalization_preserves_waveform_shape() {
        let mut samples = vec![0.1, -0.4, 0.2, -0.8, 0.6];
        let original_ratios: Vec<f32> = samples.windows(2)
            .map(|w| w[1] / w[0])
            .collect();
        
        apply_peak_normalization(&mut samples, 0.9);
        
        // Check that the relative ratios between samples are preserved
        let normalized_ratios: Vec<f32> = samples.windows(2)
            .map(|w| w[1] / w[0])
            .collect();
        
        for (orig, norm) in original_ratios.iter().zip(normalized_ratios.iter()) {
            assert!((orig - norm).abs() < 0.001, "Waveform shape not preserved");
        }
    }

    #[test]
    #[cfg(feature = "audio")]
    fn test_audio_config_normalization_defaults() {
        let config = AudioAnalysisConfig::default();
        
        assert!(config.enable_peak_normalization);
        assert_eq!(config.normalization_target, 0.95);
    }

    #[test]
    #[cfg(feature = "audio")]
    fn test_peak_normalization_integration() {
        // Test that normalization is properly integrated into the analysis pipeline
        let samples = vec![0.1; 1024]; // Simple constant signal
        let mut config = AudioAnalysisConfig::default();
        config.enable_peak_normalization = true;
        config.normalization_target = 0.8;
        
        // The function should run without panicking when normalization is enabled
        let result = analyze_with_pyin(&samples, &config);
        assert!(result.is_ok());
        
        // Disable normalization and ensure it still works
        config.enable_peak_normalization = false;
        let result2 = analyze_with_pyin(&samples, &config);
        assert!(result2.is_ok());
    }
    
    #[test]
    fn test_audio_analysis_config_defaults() {
        let config = AudioAnalysisConfig::default();
        assert_eq!(config.target_sample_rate, 44100);
        assert_eq!(config.frame_size, 2048);
        assert_eq!(config.hop_size, 512);
        assert_eq!(config.fmin, 65.0);
        assert_eq!(config.fmax, 2093.0);
        assert_eq!(config.threshold, 0.1);
        assert!(config.use_pyin_pitch_detection);
        assert_eq!(config.pyin_resolution, Some(0.05));
    }
    
    #[test]
    fn test_pitch_event_to_audio_event_conversion() {
        let pitch_event = PitchEvent {
            time: 1.0,
            frequency: 440.0,
            confidence: 0.8,
            voiced: true,
            spectral_centroid: Some(1500.0),
            harmonicity: Some(0.7),
            spectral_rolloff: Some(3000.0),
            zero_crossing_rate: Some(0.2),
        };
        
        let audio_event: AudioEvent = pitch_event.into();
        assert_eq!(audio_event.timestamp, 1.0);
        assert_eq!(audio_event.frequency, 440.0);
        assert_eq!(audio_event.amplitude, 0.8);
        assert_eq!(audio_event.channel, Some(1));
    }
}
