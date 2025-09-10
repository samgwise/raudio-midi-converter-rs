// Test file to explore pYIN crate API
#[cfg(all(feature = "audio", feature = "pyin"))]
pub fn test_pyin_api() {
    use pyin::*;
    
    // Generate a simple test signal
    let sample_rate = 44100;
    let frequency = 440.0;
    let duration = 1.0;
    let samples: Vec<f64> = (0..(sample_rate as f64 * duration) as usize)
        .map(|i| (2.0 * std::f64::consts::PI * frequency * i as f64 / sample_rate as f64).sin())
        .collect();
    
    // Create pYIN detector with proper parameters
    let fmin = 80.0;
    let fmax = 800.0;
    let frame_length = 2048;
    let win_length = None;
    let hop_length = None;
    let resolution = None;
    
    let mut pyin_exec = PYINExecutor::new(fmin, fmax, sample_rate, frame_length, win_length, hop_length, resolution);
    
    let fill_unvoiced = f64::NAN;
    let framing = Framing::Center(PadMode::Constant(0.0));
    
    // Run pYIN analysis
    let (timestamp, f0, voiced_flag, voiced_prob) = pyin_exec.pyin(&samples, fill_unvoiced, framing);
    
    println!("pYIN detector created successfully!");
    println!("  Analyzed {} frames", f0.len());
    if !f0.is_empty() {
        println!("  First frame: time={:.3}s, f0={:.1}Hz, voiced={}, prob={:.3}", 
                timestamp[0], f0[0], voiced_flag[0], voiced_prob[0]);
    }
}

#[cfg(test)]
mod tests {
    #[cfg(all(feature = "audio", feature = "pyin"))]
    #[test]
    fn explore_pyin_api() {
        use pyin::*;
        
        // Try to explore the pyin API
        let sample_rate = 44100;
        let fmin = 80.0;
        let fmax = 800.0;
        let frame_length = 2048;
        
        // Generate test audio (440 Hz sine wave)
        let samples: Vec<f64> = (0..8192)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sample_rate as f64).sin())
            .collect();
        
        // Create pYIN executor
        let mut pyin_exec = PYINExecutor::new(fmin, fmax, sample_rate, frame_length, None, None, None);
        
        let fill_unvoiced = f64::NAN;
        let framing = Framing::Center(PadMode::Constant(0.0));
        
        // Analyze
        let (timestamp, f0, voiced_flag, voiced_prob) = pyin_exec.pyin(&samples, fill_unvoiced, framing);
        
        // Verify we got results
        assert!(!f0.is_empty());
        assert_eq!(f0.len(), voiced_flag.len());
        assert_eq!(f0.len(), voiced_prob.len());
        assert_eq!(f0.len(), timestamp.len());
        
        // Should detect something close to 440 Hz in at least one frame
        let detected_440 = f0.iter().any(|&freq| !freq.is_nan() && (freq - 440.0).abs() < 50.0);
        assert!(detected_440, "Should detect frequency near 440 Hz");
        
        println!("pYIN crate API explored successfully - {} frames analyzed", f0.len());
    }
}
