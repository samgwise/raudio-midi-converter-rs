//! CSV to MIDI conversion library
//! 
//! This crate provides functionality to convert CSV audio analysis data to MIDI files.
//! The CSV format is expected to contain timing, frequency, amplitude, and channel information.

use std::io::Read;
use thiserror::Error;

pub mod midi;
pub mod parser;

pub use midi::*;
pub use parser::*;

/// Errors that can occur during CSV to MIDI conversion
#[derive(Error, Debug)]
pub enum ConversionError {
    #[error("CSV parsing error: {0}")]
    CsvError(#[from] csv::Error),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("MIDI error: {0}")]
    MidiError(#[from] midly::Error),
    
    #[error("Invalid frequency value: {0}")]
    InvalidFrequency(f64),
    
    #[error("Invalid amplitude value: {0}")]
    InvalidAmplitude(f64),
    
    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(f64),
    
    #[error("MIDI write error: {0}")]
    MidiWriteError(String),
}

/// Result type for conversion operations
pub type Result<T> = std::result::Result<T, ConversionError>;

/// Configuration for CSV to MIDI conversion
#[derive(Debug, Clone)]
pub struct ConversionConfig {
    /// MIDI ticks per quarter note (default: 480)
    pub ticks_per_quarter: u16,
    /// Minimum note duration in ticks (default: 100)
    pub min_note_duration: u32,
    /// Default velocity for notes (0-127, default: 64)
    pub default_velocity: u8,
    /// Base octave for frequency conversion (default: 4)
    pub base_octave: u8,
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self {
            ticks_per_quarter: 480,
            min_note_duration: 100,
            default_velocity: 64,
            base_octave: 4,
        }
    }
}

/// Main conversion function
/// 
/// Converts CSV audio analysis data to MIDI format
/// 
/// # Arguments
/// * `csv_data` - A reader containing CSV data
/// * `config` - Configuration for the conversion process
/// 
/// # Returns
/// * A Vec<u8> containing the MIDI file data
pub fn convert_csv_to_midi<R: Read>(
    csv_data: R,
    config: ConversionConfig,
) -> Result<Vec<u8>> {
    // Parse CSV data
    let audio_events = parse_csv_data(csv_data)?;
    
    // Convert to MIDI events
    let midi_events = convert_to_midi_events(&audio_events, &config)?;
    
    // Generate MIDI file
    generate_midi_file(midi_events, &config)
}

/// Convenience function for converting CSV string to MIDI
pub fn convert_csv_string_to_midi(
    csv_string: &str,
    config: ConversionConfig,
) -> Result<Vec<u8>> {
    convert_csv_to_midi(csv_string.as_bytes(), config)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_conversion_with_sample_data() {
        let csv_data = r#"0.241904762,0.618992,0.305124717,[2]
0.241904762,2793.83,0.305124717,[1]
0.547052154,0,0.000000000"#;
        
        let config = ConversionConfig::default();
        let result = convert_csv_string_to_midi(csv_data, config);
        
        assert!(result.is_ok(), "Conversion should succeed");
        let midi_data = result.unwrap();
        assert!(!midi_data.is_empty(), "MIDI data should not be empty");
    }
    
    #[test]
    fn test_default_config() {
        let config = ConversionConfig::default();
        assert_eq!(config.ticks_per_quarter, 480);
        assert_eq!(config.min_note_duration, 100);
        assert_eq!(config.default_velocity, 64);
        assert_eq!(config.base_octave, 4);
    }
}
