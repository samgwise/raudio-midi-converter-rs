//! pYIN-based pitch detection module
//!
//! This module provides a wrapper around the pYIN algorithm for pitch detection,
//! integrating it with our audio analysis pipeline.

use crate::Result;

#[cfg(feature = "audio")]
use pyin::{PYINExecutor, Framing, PadMode};

/// Padding mode selector
#[derive(Debug, Clone, Copy)]
pub enum PaddingMode {
    /// Zero padding
    ZeroPad,
    /// Reflection padding
    Reflect,
}

/// Configuration for pYIN pitch detection
#[derive(Debug, Clone)]
pub struct PyinConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Frame length in samples  
    pub frame_length: usize,
    /// Window length in samples (None for default: frame_length / 2)
    pub win_length: Option<usize>,
    /// Hop length in samples (None for default: frame_length / 4)
    pub hop_length: Option<usize>,
    /// Frequency resolution (None for default: 0.1)
    pub resolution: Option<f64>,
    /// Minimum frequency in Hz
    pub fmin: f64,
    /// Maximum frequency in Hz
    pub fmax: f64,
    /// Value to fill for unvoiced regions (typically f64::NAN)
    pub fill_unvoiced: f64,
    /// Enable center padding
    pub center_padding: bool,
    /// Padding mode for center padding
    pub padding_mode: PaddingMode,
}

impl Default for PyinConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            frame_length: 2048,
            win_length: None,  // Default: frame_length / 2
            hop_length: None,  // Default: frame_length / 4
            resolution: None,  // Default: 0.1
            fmin: 80.0,        // Low C2
            fmax: 2000.0,      // High C7
            fill_unvoiced: f64::NAN,
            center_padding: true,
            padding_mode: PaddingMode::ZeroPad,
        }
    }
}

/// Result of pYIN pitch analysis
#[derive(Debug, Clone)]
pub struct PyinAnalysisResult {
    /// Time stamps for each frame in seconds
    pub timestamps: Vec<f64>,
    /// Fundamental frequency estimates in Hz (NaN for unvoiced)
    pub frequencies: Vec<f64>,
    /// Voiced/unvoiced flags
    pub voiced_flags: Vec<bool>,
    /// Voiced probabilities (0.0 to 1.0)
    pub voiced_probabilities: Vec<f64>,
    /// Frame rate (frames per second)
    pub frame_rate: f64,
}

/// pYIN-based pitch detector
pub struct PyinPitchDetector {
    config: PyinConfig,
}

impl PyinPitchDetector {
    pub fn new(config: PyinConfig) -> Self {
        Self { config }
    }

    /// Analyze audio samples using pYIN algorithm
    #[cfg(feature = "audio")]
    pub fn analyze(&self, samples: &[f32]) -> Result<PyinAnalysisResult> {
        // Convert f32 to f64 for pYIN
        let samples_f64: Vec<f64> = samples.iter().map(|&x| x as f64).collect();
        
        // Create pYIN executor
        let mut pyin_exec = PYINExecutor::new(
            self.config.fmin,
            self.config.fmax,
            self.config.sample_rate,
            self.config.frame_length,
            self.config.win_length,
            self.config.hop_length,
            self.config.resolution,
        );

        // Set up framing mode
        let framing = if self.config.center_padding {
            let pad_mode = match self.config.padding_mode {
                PaddingMode::ZeroPad => PadMode::Constant(0.0),
                PaddingMode::Reflect => PadMode::Reflect,
            };
            Framing::Center(pad_mode)
        } else {
            Framing::Valid
        };

        // Run pYIN analysis
        let (timestamps, frequencies, voiced_flags, voiced_probabilities) = 
            pyin_exec.pyin(&samples_f64, self.config.fill_unvoiced, framing);

        // Calculate frame rate
        let hop_length = self.config.hop_length.unwrap_or(self.config.frame_length / 4);
        let frame_rate = self.config.sample_rate as f64 / hop_length as f64;

        Ok(PyinAnalysisResult {
            timestamps,
            frequencies,
            voiced_flags,
            voiced_probabilities,
            frame_rate,
        })
    }

    /// Analyze audio samples (stub for when audio feature is disabled)
    #[cfg(not(feature = "audio"))]
    pub fn analyze(&self, _samples: &[f32]) -> Result<PyinAnalysisResult> {
        Err(ConversionError::MidiWriteError(
            "Audio analysis feature is not enabled".to_string()
        ))
    }
}

/// Create a pYIN detector with sensible defaults for audio analysis
pub fn create_default_pyin_detector(sample_rate: u32, fmin: f64, fmax: f64) -> PyinPitchDetector {
    let config = PyinConfig {
        sample_rate,
        fmin,
        fmax,
        ..Default::default()
    };
    PyinPitchDetector::new(config)
}

/// Create a pYIN detector optimized for musical pitch detection
pub fn create_musical_pyin_detector(sample_rate: u32) -> PyinPitchDetector {
    let config = PyinConfig {
        sample_rate,
        frame_length: 2048,
        hop_length: Some(512),  // 25% overlap
        fmin: 65.0,            // C2
        fmax: 2093.0,          // C7
        resolution: Some(0.05), // Higher resolution for music
        ..Default::default()
    };
    PyinPitchDetector::new(config)
}

/// Create a pYIN detector optimized for speech analysis  
pub fn create_speech_pyin_detector(sample_rate: u32) -> PyinPitchDetector {
    let config = PyinConfig {
        sample_rate,
        frame_length: 1024,     // Shorter frames for speech
        hop_length: Some(256),  // More overlap for speech
        fmin: 80.0,            // Lower bound for speech F0
        fmax: 400.0,           // Upper bound for speech F0
        resolution: Some(0.1),  // Standard resolution
        ..Default::default()
    };
    PyinPitchDetector::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    #[cfg(feature = "audio")]
    fn test_pyin_detector_sine_wave() {
        let sample_rate = 44100;
        let frequency = 440.0;
        let duration = 0.5; // 0.5 seconds
        
        // Generate sine wave
        let samples: Vec<f32> = (0..(sample_rate as f64 * duration) as usize)
            .map(|i| (2.0 * PI * frequency * i as f64 / sample_rate as f64).sin() as f32)
            .collect();

        let detector = create_musical_pyin_detector(sample_rate);
        let result = detector.analyze(&samples).unwrap();
        
        // Should detect frames
        assert!(!result.frequencies.is_empty());
        assert_eq!(result.frequencies.len(), result.timestamps.len());
        assert_eq!(result.frequencies.len(), result.voiced_flags.len());
        assert_eq!(result.frequencies.len(), result.voiced_probabilities.len());

        // Should detect frequency close to 440 Hz in some frames
        let detected_440 = result.frequencies.iter()
            .zip(result.voiced_flags.iter())
            .any(|(&freq, &voiced)| voiced && !freq.is_nan() && (freq - frequency).abs() < 10.0);
        
        assert!(detected_440, "Should detect 440 Hz frequency");
        
        // Should have reasonable frame rate
        assert!(result.frame_rate > 80.0 && result.frame_rate < 200.0);
    }

    #[test]
    #[cfg(feature = "audio")]
    fn test_pyin_detector_noise() {
        use rand::Rng;
        let sample_rate = 44100;
        
        // Generate noise
        let mut rng = rand::thread_rng();
        let samples: Vec<f32> = (0..8192)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        let detector = create_default_pyin_detector(sample_rate, 80.0, 800.0);
        let result = detector.analyze(&samples).unwrap();
        
        // Should analyze frames
        assert!(!result.frequencies.is_empty());
        
        // Most frames should be unvoiced for pure noise
        let voiced_count = result.voiced_flags.iter().filter(|&&v| v).count();
        let total_frames = result.voiced_flags.len();
        
        assert!(voiced_count < total_frames / 2, 
            "Too many voiced frames ({}/{}) detected in noise", 
            voiced_count, total_frames);
    }

    #[test]
    fn test_pyin_config_defaults() {
        let config = PyinConfig::default();
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.frame_length, 2048);
        assert!(config.win_length.is_none());
        assert!(config.hop_length.is_none());
        assert!(config.resolution.is_none());
        assert_eq!(config.fmin, 80.0);
        assert_eq!(config.fmax, 2000.0);
        assert!(config.fill_unvoiced.is_nan());
        assert!(config.center_padding);
    }

    #[test]  
    fn test_pyin_detector_creation() {
        let detector = create_musical_pyin_detector(48000);
        assert_eq!(detector.config.sample_rate, 48000);
        assert_eq!(detector.config.fmin, 65.0);
        assert_eq!(detector.config.fmax, 2093.0);
        
        let speech_detector = create_speech_pyin_detector(16000);
        assert_eq!(speech_detector.config.sample_rate, 16000);
        assert_eq!(speech_detector.config.fmin, 80.0);
        assert_eq!(speech_detector.config.fmax, 400.0);
    }
}
