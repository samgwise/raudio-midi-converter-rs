//! CSV parsing module for audio analysis data
//!
//! This module handles parsing of CSV data containing audio analysis information
//! with timestamps, frequencies, amplitudes, and channel indicators.

use crate::{ConversionError, Result};
use csv::ReaderBuilder;
use std::io::Read;

/// Represents a single audio event from the CSV data
#[derive(Debug, Clone, PartialEq)]
pub struct AudioEvent {
    pub line_number: u32,
    pub timestamp: f64,    // Time in seconds
    pub frequency: f64,    // Frequency in Hz (0 indicates silence)
    pub amplitude: f64,    // Amplitude 0.0-1.0
    pub channel: Option<u8>, // Channel number (1 or 2, None for silence)
}

impl AudioEvent {
    /// Returns true if this event represents silence (frequency == 0)
    pub fn is_silence(&self) -> bool {
        self.frequency == 0.0
    }
    
    /// Returns true if this event has sound (frequency > 0)
    pub fn has_sound(&self) -> bool {
        !self.is_silence()
    }
    
    /// Validates that the event has reasonable values
    pub fn validate(&self) -> Result<()> {
        if self.timestamp < 0.0 {
            return Err(ConversionError::InvalidTimestamp(self.timestamp));
        }
        
        if self.frequency < 0.0 {
            return Err(ConversionError::InvalidFrequency(self.frequency));
        }
        
        if self.amplitude < 0.0 || self.amplitude > 1.0 {
            return Err(ConversionError::InvalidAmplitude(self.amplitude));
        }
        
        Ok(())
    }
}

/// Parses CSV data into a vector of AudioEvent structures
/// 
/// The expected CSV format is:
/// timestamp,frequency,amplitude,[channel]
/// 
/// Example:
/// 0.241904762,0.618992,0.305124717,[2]
/// 0.241904762,2793.83,0.305124717,[1]
/// 0.547052154,0,0.000000000
pub fn parse_csv_data<R: Read>(reader: R) -> Result<Vec<AudioEvent>> {
    let mut csv_reader = ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)  // Allow variable number of fields per record
        .from_reader(reader);
    
    let mut events = Vec::new();
    let mut line_number = 1;
    
    for result in csv_reader.records() {
        let record = result?;
        
        // Parse all fields from the record
        if record.len() < 3 {
            return Err(ConversionError::InvalidFrequency(-1.0));
        }
        
        let timestamp: f64 = record.get(0)
            .ok_or_else(|| ConversionError::InvalidTimestamp(-1.0))?
            .parse()
            .map_err(|_| ConversionError::InvalidTimestamp(-1.0))?;
        
        let frequency: f64 = record.get(1)
            .ok_or_else(|| ConversionError::InvalidFrequency(-1.0))?
            .parse()
            .map_err(|_| ConversionError::InvalidFrequency(-1.0))?;
        
        let amplitude: f64 = record.get(2)
            .ok_or_else(|| ConversionError::InvalidAmplitude(-1.0))?
            .parse()
            .map_err(|_| ConversionError::InvalidAmplitude(-1.0))?;
        
        // Parse channel if present (format: [1] or [2])
        let channel = if record.len() > 3 {
            parse_channel(record.get(3).unwrap_or(""))?
        } else {
            None
        };
        
        let event = AudioEvent {
            line_number,
            timestamp,
            frequency,
            amplitude,
            channel,
        };
        
        event.validate()?;
        events.push(event);
        line_number += 1;
    }
    
    Ok(events)
}


/// Parses channel information from format "[1]" or "[2]"
fn parse_channel(channel_str: &str) -> Result<Option<u8>> {
    let trimmed = channel_str.trim();
    
    if trimmed.is_empty() {
        return Ok(None);
    }
    
    // Handle format [1] or [2]
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        let inner = &trimmed[1..trimmed.len()-1];
        let channel_num: u8 = inner.parse()
            .map_err(|_| ConversionError::InvalidFrequency(-1.0))?;
        
        Ok(Some(channel_num))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_simple_data() {
        let csv_data = "0.241904762,0.618992,0.305124717,[2]\n0.241904762,2793.83,0.305124717,[1]\n0.547052154,0,0.000000000";
        
        let events = parse_csv_data(csv_data.as_bytes()).unwrap();
        
        assert_eq!(events.len(), 3);
        
        assert_eq!(events[0].line_number, 1);
        assert_eq!(events[0].timestamp, 0.241904762);
        assert_eq!(events[0].frequency, 0.618992);
        assert_eq!(events[0].amplitude, 0.305124717);
        assert_eq!(events[0].channel, Some(2));
        assert!(!events[0].is_silence());
        
        assert_eq!(events[1].channel, Some(1));
        assert_eq!(events[1].frequency, 2793.83);
        
        assert!(events[2].is_silence());
        assert_eq!(events[2].channel, None);
    }
    
    #[test]
    fn test_parse_channel() {
        assert_eq!(parse_channel("[1]").unwrap(), Some(1));
        assert_eq!(parse_channel("[2]").unwrap(), Some(2));
        assert_eq!(parse_channel("").unwrap(), None);
        assert_eq!(parse_channel("   ").unwrap(), None);
    }
    
    #[test]
    fn test_audio_event_validation() {
        let valid_event = AudioEvent {
            line_number: 1,
            timestamp: 1.0,
            frequency: 440.0,
            amplitude: 0.5,
            channel: Some(1),
        };
        assert!(valid_event.validate().is_ok());
        
        let invalid_timestamp = AudioEvent {
            line_number: 1,
            timestamp: -1.0,
            frequency: 440.0,
            amplitude: 0.5,
            channel: Some(1),
        };
        assert!(invalid_timestamp.validate().is_err());
        
        let invalid_amplitude = AudioEvent {
            line_number: 1,
            timestamp: 1.0,
            frequency: 440.0,
            amplitude: 1.5,
            channel: Some(1),
        };
        assert!(invalid_amplitude.validate().is_err());
    }
}
