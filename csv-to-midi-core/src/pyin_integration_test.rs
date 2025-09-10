//! Integration test for pYIN pitch detection in audio analysis
//!
//! This module tests that the pYIN algorithm integrates correctly with the audio analysis pipeline.

#[cfg(test)]
mod tests {
    use super::super::*;
    use std::f64::consts::PI;

    #[test]
    #[cfg(feature = "audio")]
    fn test_pyin_integration_with_audio_analysis() {
        // Generate a test signal with two distinct frequencies
        let sample_rate = 44100;
        let duration = 1.0; // 1 second
        let f1 = 220.0; // A3
        let f2 = 440.0; // A4

        let samples: Vec<f32> = (0..(sample_rate as f64 * duration) as usize)
            .map(|i| {
                let t = i as f64 / sample_rate as f64;
                if t < 0.5 {
                    // First half: 220 Hz
                    (2.0 * PI * f1 * t).sin() as f32
                } else {
                    // Second half: 440 Hz
                    (2.0 * PI * f2 * t).sin() as f32
                }
            })
            .collect();

        // Test with pYIN enabled (default)
        let pyin_config = AudioAnalysisConfig {
            target_sample_rate: sample_rate,
            frame_size: 2048,
            hop_size: 512,
            fmin: 100.0,
            fmax: 800.0,
            threshold: 0.1,
            use_pyin_pitch_detection: true,
            pyin_resolution: Some(0.05),
            ..Default::default()
        };

        let pyin_result = crate::audio::analyze_with_pyin(&samples, &pyin_config).unwrap();

        // Test with autocorrelation fallback
        let autocorr_config = AudioAnalysisConfig {
            use_pyin_pitch_detection: false,
            ..pyin_config.clone()
        };

        let autocorr_result = crate::audio::analyze_with_pyin(&samples, &autocorr_config).unwrap();

        // Verify both methods produce results
        assert!(!pyin_result.pitch_events.is_empty(), "pYIN should produce pitch events");
        assert!(!autocorr_result.pitch_events.is_empty(), "Autocorrelation should produce pitch events");

        // pYIN should generally be more accurate, let's check that it detects the frequencies
        let pyin_freqs: Vec<f64> = pyin_result.pitch_events
            .iter()
            .filter_map(|event| if event.voiced { Some(event.frequency) } else { None })
            .collect();

        // Should detect frequencies close to 220 Hz and 440 Hz
        let detects_220 = pyin_freqs.iter().any(|&freq| (freq - f1).abs() < 20.0);
        let detects_440 = pyin_freqs.iter().any(|&freq| (freq - f2).abs() < 20.0);

        assert!(detects_220 || detects_440, 
            "pYIN should detect at least one of the test frequencies. Detected: {:?}", 
            pyin_freqs.iter().take(10).collect::<Vec<_>>());

        println!("pYIN integration test successful!");
        println!("  pYIN events: {}, Autocorr events: {}", 
                 pyin_result.pitch_events.len(), 
                 autocorr_result.pitch_events.len());
        println!("  pYIN voiced frames: {}, Autocorr voiced frames: {}", 
                 pyin_result.pitch_events.iter().filter(|e| e.voiced).count(),
                 autocorr_result.pitch_events.iter().filter(|e| e.voiced).count());
    }

    #[test]
    #[cfg(feature = "audio")]
    fn test_pyin_vs_autocorr_consistency() {
        // Test that both methods handle the same input and produce reasonable results
        let sample_rate = 44100;
        
        // Generate pure tone
        let frequency = 330.0; // E4
        let samples: Vec<f32> = (0..8192)
            .map(|i| (2.0 * PI * frequency * i as f64 / sample_rate as f64).sin() as f32)
            .collect();

        let base_config = AudioAnalysisConfig {
            target_sample_rate: sample_rate,
            frame_size: 2048,
            hop_size: 512,
            fmin: 200.0,
            fmax: 500.0,
            threshold: 0.1,
            enable_spectral_analysis: false, // Simplify for this test
            enable_harmonicity_analysis: false,
            enable_zero_crossing_analysis: false,
            ..Default::default()
        };

        // Test pYIN method
        let pyin_config = AudioAnalysisConfig {
            use_pyin_pitch_detection: true,
            ..base_config.clone()
        };
        let pyin_result = crate::audio::analyze_with_pyin(&samples, &pyin_config).unwrap();

        // Test autocorrelation method
        let autocorr_config = AudioAnalysisConfig {
            use_pyin_pitch_detection: false,
            ..base_config
        };
        let autocorr_result = crate::audio::analyze_with_pyin(&samples, &autocorr_config).unwrap();

        // Both should detect the frequency (within some tolerance)
        let pyin_avg_freq = pyin_result.pitch_events
            .iter()
            .filter(|e| e.voiced && e.frequency > 0.0)
            .map(|e| e.frequency)
            .sum::<f64>() / pyin_result.pitch_events.iter().filter(|e| e.voiced).count() as f64;

        let autocorr_avg_freq = autocorr_result.pitch_events
            .iter()
            .filter(|e| e.voiced && e.frequency > 0.0)
            .map(|e| e.frequency)
            .sum::<f64>() / autocorr_result.pitch_events.iter().filter(|e| e.voiced).count() as f64;

        println!("Frequency detection comparison:");
        println!("  Target: {:.1} Hz", frequency);
        println!("  pYIN average: {:.1} Hz", pyin_avg_freq);
        println!("  Autocorr average: {:.1} Hz", autocorr_avg_freq);

        // Both should be reasonably close to the target frequency
        assert!((pyin_avg_freq - frequency).abs() < 50.0, 
            "pYIN frequency too far from target: {:.1} vs {:.1}", pyin_avg_freq, frequency);
        assert!((autocorr_avg_freq - frequency).abs() < 50.0, 
            "Autocorr frequency too far from target: {:.1} vs {:.1}", autocorr_avg_freq, frequency);
    }
}
