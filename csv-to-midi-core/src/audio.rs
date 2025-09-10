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
    rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction}
};

/// Configuration specific to audio analysis
#[derive(Debug, Clone)]
pub struct AudioAnalysisConfig {
    /// Sample rate for analysis (PYIN works best at 44.1kHz)
    pub target_sample_rate: u32,
    /// Frame size for PYIN analysis in samples
    pub frame_size: usize,
    /// Hop size for PYIN analysis in samples (frame overlap)
    pub hop_size: usize,
    /// Minimum frequency to detect in Hz
    pub fmin: f64,
    /// Maximum frequency to detect in Hz
    pub fmax: f64,
    /// Threshold for voiced/unvoiced detection (0.0-1.0)
    pub threshold: f64,
}

impl Default for AudioAnalysisConfig {
    fn default() -> Self {
        Self {
            target_sample_rate: 44100,
            frame_size: 2048,
            hop_size: 512,  // 25% overlap
            fmin: 65.0,     // C2
            fmax: 2093.0,   // C7
            threshold: 0.1, // PYIN default
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

/// A single pitch detection event
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
/// Analyze audio samples using simple autocorrelation-based pitch detection
/// Note: This is a simplified implementation. For production use, consider more sophisticated algorithms like PYIN
fn analyze_with_pyin(
    samples: &[f32],
    config: &AudioAnalysisConfig,
) -> Result<PitchAnalysisResult> {
    let sample_rate = config.target_sample_rate;
    let duration = samples.len() as f64 / sample_rate as f64;
    
    // Simple frame-based pitch detection using autocorrelation
    let mut pitch_events = Vec::new();
    let frame_size = config.frame_size;
    let hop_size = config.hop_size;
    
    // Process audio in overlapping frames
    for frame_start in (0..samples.len()).step_by(hop_size) {
        if frame_start + frame_size > samples.len() {
            break;
        }
        
        let frame = &samples[frame_start..frame_start + frame_size];
        let time = frame_start as f64 / sample_rate as f64;
        
        // Simple autocorrelation-based pitch detection
        let (frequency, confidence) = estimate_pitch_autocorr(frame, sample_rate, config.fmin, config.fmax);
        let voiced = frequency > 0.0 && confidence > config.threshold;
        
        pitch_events.push(PitchEvent {
            time,
            frequency: if voiced { frequency } else { 0.0 },
            confidence,
            voiced,
        });
    }
    
    Ok(PitchAnalysisResult {
        pitch_events,
        sample_rate,
        duration,
    })
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

/// Convert audio analysis results to AudioEvent format for MIDI conversion
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

/// Convert audio file directly to MIDI using PYIN analysis
pub fn convert_audio_to_midi<P: AsRef<Path>>(
    file_path: P,
    audio_config: &AudioAnalysisConfig,
    midi_config: &ConversionConfig,
) -> Result<Vec<u8>> {
    // Analyze audio file
    let analysis = analyze_audio_file(file_path, audio_config)?;
    
    // Convert to audio events with minimum note duration based on hop size
    let min_note_duration = (audio_config.hop_size as f64 * 2.0) / audio_config.target_sample_rate as f64;
    let audio_events = pitch_analysis_to_audio_events(&analysis, min_note_duration);
    
    // Convert to MIDI using existing pipeline
    let midi_events = crate::midi::convert_to_midi_events(&audio_events, midi_config)?;
    crate::midi::generate_midi_file(midi_events, midi_config)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_audio_analysis_config_defaults() {
        let config = AudioAnalysisConfig::default();
        assert_eq!(config.target_sample_rate, 44100);
        assert_eq!(config.frame_size, 2048);
        assert_eq!(config.hop_size, 512);
        assert_eq!(config.fmin, 65.0);
        assert_eq!(config.fmax, 2093.0);
        assert_eq!(config.threshold, 0.1);
    }
    
    #[test]
    fn test_pitch_event_to_audio_event_conversion() {
        let pitch_event = PitchEvent {
            time: 1.0,
            frequency: 440.0,
            confidence: 0.8,
            voiced: true,
        };
        
        let audio_event: AudioEvent = pitch_event.into();
        assert_eq!(audio_event.timestamp, 1.0);
        assert_eq!(audio_event.frequency, 440.0);
        assert_eq!(audio_event.amplitude, 0.8);
        assert_eq!(audio_event.channel, Some(1));
    }
}
