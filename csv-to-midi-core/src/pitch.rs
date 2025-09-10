//! Enhanced pitch detection module
//!
//! This module implements improved pitch detection algorithms inspired by pYIN
//! and research findings comparing various pitch detection approaches.
//!
//! Key improvements over basic autocorrelation:
//! - Probabilistic pitch candidate estimation
//! - Multiple pitch candidate tracking
//! - Improved difference function with proper normalization
//! - Parabolic interpolation for sub-sample accuracy
//! - Better handling of pitch doubling/halving

use crate::{Result, ConversionError};
use std::f64::consts::PI;

/// Configuration for enhanced pitch detection
#[derive(Debug, Clone)]
pub struct PitchDetectionConfig {
    /// Sample rate in Hz
    pub sample_rate: f64,
    /// Frame size in samples
    pub frame_size: usize,
    /// Hop size in samples
    pub hop_size: usize,
    /// Minimum fundamental frequency in Hz
    pub fmin: f64,
    /// Maximum fundamental frequency in Hz
    pub fmax: f64,
    /// Threshold for voicing detection (0.0-1.0)
    pub voiced_threshold: f64,
    /// Number of pitch candidates to consider
    pub num_candidates: usize,
    /// Enable probabilistic smoothing
    pub enable_smoothing: bool,
    /// Beta parameter for YIN difference function
    pub beta: f64,
}

impl Default for PitchDetectionConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100.0,
            frame_size: 2048,
            hop_size: 512,
            fmin: 65.0,
            fmax: 2093.0,
            voiced_threshold: 0.1,
            num_candidates: 5,
            enable_smoothing: true,
            beta: 0.02,
        }
    }
}

/// Pitch candidate with probability
#[derive(Debug, Clone)]
pub struct PitchCandidate {
    /// Frequency in Hz (0.0 if unvoiced)
    pub frequency: f64,
    /// Confidence/probability (0.0-1.0)
    pub probability: f64,
    /// Whether this candidate represents voiced speech/audio
    pub voiced: bool,
}

/// Result of pitch analysis for a single frame
#[derive(Debug, Clone)]
pub struct PitchAnalysisFrame {
    /// Time offset in seconds
    pub time: f64,
    /// Best pitch candidate
    pub best_pitch: PitchCandidate,
    /// All pitch candidates (sorted by probability)
    pub candidates: Vec<PitchCandidate>,
}

/// Enhanced pitch detector based on pYIN methodology
pub struct EnhancedPitchDetector {
    config: PitchDetectionConfig,
    previous_frames: Vec<PitchAnalysisFrame>,
}

impl EnhancedPitchDetector {
    pub fn new(config: PitchDetectionConfig) -> Self {
        Self {
            config,
            previous_frames: Vec::new(),
        }
    }

    /// Analyze audio samples for pitch content
    pub fn analyze(&mut self, samples: &[f32]) -> Result<Vec<PitchAnalysisFrame>> {
        let mut results = Vec::new();
        
        // Process audio in overlapping frames
        for frame_start in (0..samples.len()).step_by(self.config.hop_size) {
            if frame_start + self.config.frame_size > samples.len() {
                break;
            }
            
            let frame = &samples[frame_start..frame_start + self.config.frame_size];
            let time = frame_start as f64 / self.config.sample_rate;
            
            let analysis_frame = self.analyze_frame(frame, time)?;
            results.push(analysis_frame);
        }
        
        // Apply temporal smoothing if enabled
        if self.config.enable_smoothing {
            self.apply_temporal_smoothing(&mut results)?;
        }
        
        // Store for future smoothing
        self.previous_frames.extend_from_slice(&results);
        
        // Keep only recent frames for memory efficiency
        if self.previous_frames.len() > 100 {
            self.previous_frames.drain(0..self.previous_frames.len() - 50);
        }
        
        Ok(results)
    }

    /// Analyze a single audio frame
    fn analyze_frame(&self, frame: &[f32], time: f64) -> Result<PitchAnalysisFrame> {
        // Convert to f64 for better precision
        let frame_f64: Vec<f64> = frame.iter().map(|&x| x as f64).collect();
        
        // Apply window function (Hamming window)
        let windowed_frame = self.apply_window(&frame_f64);
        
        // Compute enhanced difference function (YIN-style)
        let difference_function = self.compute_difference_function(&windowed_frame)?;
        
        // Find pitch candidates
        let candidates = self.find_pitch_candidates(&difference_function)?;
        
        // Select best candidate
        let best_pitch = candidates.first()
            .cloned()
            .unwrap_or(PitchCandidate {
                frequency: 0.0,
                probability: 0.0,
                voiced: false,
            });
        
        Ok(PitchAnalysisFrame {
            time,
            best_pitch,
            candidates,
        })
    }
    
    /// Apply Hamming window to reduce spectral leakage
    fn apply_window(&self, frame: &[f64]) -> Vec<f64> {
        let mut windowed = Vec::with_capacity(frame.len());
        let n = frame.len() as f64;
        
        for (i, &sample) in frame.iter().enumerate() {
            let window_value = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1.0)).cos();
            windowed.push(sample * window_value);
        }
        
        windowed
    }
    
    /// Compute YIN-style difference function with improvements
    fn compute_difference_function(&self, frame: &[f64]) -> Result<Vec<f64>> {
        let max_lag = (self.config.sample_rate / self.config.fmin) as usize;
        let min_lag = (self.config.sample_rate / self.config.fmax) as usize;
        let max_lag = max_lag.min(frame.len() / 2);
        
        if min_lag >= max_lag {
            return Err(ConversionError::InvalidFrequency(self.config.fmin));
        }
        
        let mut difference_function = vec![0.0; max_lag + 1];
        
        // Compute difference function
        for lag in min_lag..=max_lag {
            let mut diff = 0.0;
            for i in 0..(frame.len() - lag) {
                let delta = frame[i] - frame[i + lag];
                diff += delta * delta;
            }
            difference_function[lag] = diff;
        }
        
        // Cumulative mean normalized difference function (CMND)
        let mut cmnd = vec![1.0; max_lag + 1];
        cmnd[0] = 1.0;
        
        let mut running_sum = 0.0;
        for lag in 1..=max_lag {
            running_sum += difference_function[lag];
            if running_sum > 0.0 {
                cmnd[lag] = difference_function[lag] / (running_sum / lag as f64);
            }
        }
        
        Ok(cmnd)
    }
    
    /// Find multiple pitch candidates from difference function
    fn find_pitch_candidates(&self, cmnd: &[f64]) -> Result<Vec<PitchCandidate>> {
        let mut candidates = Vec::new();
        
        let min_lag = (self.config.sample_rate / self.config.fmax) as usize;
        let max_lag = (self.config.sample_rate / self.config.fmin) as usize;
        let max_lag = max_lag.min(cmnd.len() - 1);
        
        // Find local minima in the CMND function
        for lag in (min_lag + 1)..(max_lag - 1) {
            if cmnd[lag] < cmnd[lag - 1] && cmnd[lag] < cmnd[lag + 1] {
                let confidence = 1.0 - cmnd[lag];
                
                if confidence > 0.1 { // Minimum confidence threshold
                    // Parabolic interpolation for sub-sample accuracy
                    let refined_lag = self.parabolic_interpolation(cmnd, lag);
                    let frequency = self.config.sample_rate / refined_lag;
                    
                    // Check if frequency is in valid range
                    if frequency >= self.config.fmin && frequency <= self.config.fmax {
                        candidates.push(PitchCandidate {
                            frequency,
                            probability: confidence,
                            voiced: confidence > self.config.voiced_threshold,
                        });
                    }
                }
            }
        }
        
        // Sort by probability (highest first)
        candidates.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());
        
        // Keep only top candidates
        candidates.truncate(self.config.num_candidates);
        
        // Add unvoiced candidate if no strong voiced candidates
        if candidates.is_empty() || candidates[0].probability < self.config.voiced_threshold {
            candidates.insert(0, PitchCandidate {
                frequency: 0.0,
                probability: 1.0 - candidates.get(0).map_or(0.0, |c| c.probability),
                voiced: false,
            });
        }
        
        Ok(candidates)
    }
    
    /// Parabolic interpolation for sub-sample accuracy
    fn parabolic_interpolation(&self, data: &[f64], peak_index: usize) -> f64 {
        if peak_index == 0 || peak_index >= data.len() - 1 {
            return peak_index as f64;
        }
        
        let y1 = data[peak_index - 1];
        let y2 = data[peak_index];
        let y3 = data[peak_index + 1];
        
        let a = (y1 - 2.0 * y2 + y3) / 2.0;
        
        if a.abs() < 1e-10 {
            return peak_index as f64;
        }
        
        let x_offset = -(y3 - y1) / (4.0 * a);
        peak_index as f64 + x_offset
    }
    
    /// Apply temporal smoothing using Hidden Markov Model-like approach
    fn apply_temporal_smoothing(&self, frames: &mut [PitchAnalysisFrame]) -> Result<()> {
        if frames.len() < 2 {
            return Ok(());
        }
        
        // Simple smoothing: favor candidates that are close to previous frame
        for i in 1..frames.len() {
            let prev_freq = frames[i - 1].best_pitch.frequency;
            
            // Find candidate closest to previous frequency
            let mut best_candidate = frames[i].candidates[0].clone();
            let mut best_score = f64::INFINITY;
            
            for candidate in &frames[i].candidates {
                if candidate.voiced && prev_freq > 0.0 {
                    // Frequency continuity score
                    let freq_diff = (candidate.frequency - prev_freq).abs() / prev_freq;
                    let continuity_score = freq_diff;
                    let combined_score = continuity_score - candidate.probability * 0.5;
                    
                    if combined_score < best_score {
                        best_score = combined_score;
                        best_candidate = candidate.clone();
                    }
                } else if !candidate.voiced && prev_freq == 0.0 {
                    // Prefer unvoiced if previous was unvoiced
                    best_candidate = candidate.clone();
                    break;
                }
            }
            
            frames[i].best_pitch = best_candidate;
        }
        
        Ok(())
    }
}

/// Convenience function for basic pitch detection
pub fn detect_pitch_enhanced(
    samples: &[f32],
    config: PitchDetectionConfig,
) -> Result<Vec<PitchAnalysisFrame>> {
    let mut detector = EnhancedPitchDetector::new(config);
    detector.analyze(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_pitch_detection() {
        // Generate test sine wave at 440 Hz
        let sample_rate = 44100.0;
        let frequency = 440.0;
        let duration = 1.0; // 1 second
        let samples: Vec<f32> = (0..(sample_rate * duration) as usize)
            .map(|i| (2.0 * PI * frequency * i as f64 / sample_rate).sin() as f32)
            .collect();
        
        let config = PitchDetectionConfig {
            sample_rate,
            fmin: 80.0,
            fmax: 800.0,
            ..Default::default()
        };
        
        let results = detect_pitch_enhanced(&samples, config).unwrap();
        
        // Should detect frequency close to 440 Hz
        assert!(!results.is_empty());
        let detected_freq = results[10].best_pitch.frequency; // Skip first few frames
        assert!((detected_freq - frequency).abs() < 5.0, "Expected ~440 Hz, got {}", detected_freq);
        assert!(results[10].best_pitch.voiced, "Should be detected as voiced");
    }

    #[test]
    fn test_unvoiced_detection() {
        // Generate white noise
        let samples: Vec<f32> = (0..44100)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();
        
        let config = PitchDetectionConfig::default();
        let results = detect_pitch_enhanced(&samples, config).unwrap();
        
        // Most frames should be unvoiced
        let voiced_count = results.iter().filter(|f| f.best_pitch.voiced).count();
        assert!(voiced_count < results.len() / 4, "Too many frames detected as voiced in noise");
    }
}
