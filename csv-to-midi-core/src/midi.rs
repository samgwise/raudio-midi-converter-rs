//! MIDI conversion and generation module
//!
//! This module handles converting parsed audio events into MIDI events
//! and generating the final MIDI file format.

use crate::{parser::AudioEvent, ConversionConfig, Result};
use midly::{Format, Header, MidiMessage, Timing, Track, TrackEvent, TrackEventKind};
use std::collections::HashMap;

/// Represents a MIDI note event with timing information
#[derive(Debug, Clone)]
pub struct MidiNoteEvent {
    pub channel: u8,
    pub note: u8,
    pub velocity: u8,
    pub start_tick: u32,
    pub duration_ticks: u32,
}

/// Represents a MIDI Control Change event
#[derive(Debug, Clone)]
pub struct MidiCCEvent {
    pub channel: u8,
    pub controller: u8,
    pub value: u8,
    pub tick: u32,
}

/// Combined MIDI events structure
#[derive(Debug, Clone)]
pub struct MidiEventCollection {
    pub note_events: Vec<MidiNoteEvent>,
    pub cc_events: Vec<MidiCCEvent>,
}

/// Convert frequency in Hz to MIDI note number
/// Uses equal temperament with A4 = 440Hz = MIDI note 69
pub fn frequency_to_midi_note(frequency: f64) -> u8 {
    if frequency <= 0.0 {
        return 0; // Silence
    }
    
    // MIDI note formula: note = 69 + 12 * log2(frequency / 440)
    let note_float = 69.0 + 12.0 * (frequency / 440.0).log2();
    
    // Clamp to valid MIDI range (0-127)
    note_float.round().max(0.0).min(127.0) as u8
}

/// Convert amplitude (0.0-1.0) to MIDI velocity (0-127)
pub fn amplitude_to_velocity(amplitude: f64, default_velocity: u8) -> u8 {
    if amplitude <= 0.0 {
        return 0;
    }
    
    // Scale amplitude to velocity range, with minimum velocity for audible notes
    let velocity = (amplitude * 127.0).round() as u8;
    velocity.max(1).min(127).max(default_velocity / 4)
}

/// Convert timestamp in seconds to MIDI ticks
pub fn seconds_to_ticks(seconds: f64, ticks_per_quarter: u16) -> u32 {
    // Assuming 120 BPM (2 beats per second)
    // 1 quarter note = 0.5 seconds at 120 BPM
    let quarters_per_second = 2.0;
    let ticks = seconds * quarters_per_second * ticks_per_quarter as f64;
    ticks.round().max(0.0) as u32
}

/// Convert frequency to pitch contour CC value (0-127)
/// Maps frequency to a CC range where 64 = center/no bend
pub fn frequency_to_pitch_contour_cc(frequency: f64) -> u8 {
    if frequency <= 0.0 {
        return 64; // Center value for silence
    }
    
    // Convert frequency to MIDI note number (can be fractional)
    let note_float = 69.0 + 12.0 * (frequency / 440.0).log2();
    
    // Get fractional part for fine pitch control
    let fractional_part = note_float.fract();
    
    // Map fractional part to CC range (0-127)
    // 0.0 = note is perfectly on pitch = CC 64 (center)
    // -0.5 = half step below = CC 0
    // +0.5 = half step above = CC 127
    let cc_value = 64.0 + (fractional_part * 127.0);
    cc_value.round().clamp(0.0, 127.0) as u8
}

/// Convert amplitude (0.0-1.0) to CC value (0-127)
pub fn amplitude_to_cc_value(amplitude: f64) -> u8 {
    if amplitude <= 0.0 {
        return 0;
    }
    
    let cc_value = (amplitude * 127.0).round();
    cc_value.clamp(0.0, 127.0) as u8
}

/// Convert audio events to MIDI note events
pub fn convert_to_midi_events(
    audio_events: &[AudioEvent],
    config: &ConversionConfig,
) -> Result<Vec<MidiNoteEvent>> {
    let collection = convert_to_midi_events_with_cc(audio_events, config)?;
    Ok(collection.note_events)
}

/// Convert audio events to MIDI events including notes and CC data
pub fn convert_to_midi_events_with_cc(
    audio_events: &[AudioEvent],
    config: &ConversionConfig,
) -> Result<MidiEventCollection> {
    let mut note_events = Vec::new();
    let mut cc_events = Vec::new();
    let mut active_notes: HashMap<(u8, u8), (u32, u8)> = HashMap::new(); // (channel, note) -> (start_tick, velocity)
    
    for event in audio_events {
        let tick = seconds_to_ticks(event.timestamp, config.ticks_per_quarter);
        let channel = event.channel.unwrap_or(1) - 1; // Convert to 0-based channel
        
        if event.is_silence() {
            // End all active notes when we hit silence
            for ((channel, note), (start_tick, velocity)) in active_notes.drain() {
                let duration = tick.saturating_sub(start_tick).max(config.min_note_duration);
                
                note_events.push(MidiNoteEvent {
                    channel,
                    note,
                    velocity,
                    start_tick,
                    duration_ticks: duration,
                });
            }
            
            // Add CC events for silence (reset values)
            cc_events.push(MidiCCEvent {
                channel,
                controller: 100, // Pitch contour
                value: 64,       // Center value (no pitch bend)
                tick,
            });
            cc_events.push(MidiCCEvent {
                channel,
                controller: 101, // Amplitude
                value: 0,        // Silence
                tick,
            });
            
            continue;
        }
        
        let midi_note = frequency_to_midi_note(event.frequency);
        if midi_note == 0 {
            continue; // Skip invalid frequencies
        }
        
        let velocity = amplitude_to_velocity(event.amplitude, config.default_velocity);
        
        // Generate CC events for pitch contour and amplitude
        let pitch_contour_cc = frequency_to_pitch_contour_cc(event.frequency);
        let amplitude_cc = amplitude_to_cc_value(event.amplitude);
        
        cc_events.push(MidiCCEvent {
            channel,
            controller: 100, // Pitch contour
            value: pitch_contour_cc,
            tick,
        });
        
        cc_events.push(MidiCCEvent {
            channel,
            controller: 101, // Amplitude
            value: amplitude_cc,
            tick,
        });
        
        let key = (channel, midi_note);
        
        // If this note is already playing, end the previous instance
        if let Some((start_tick, prev_velocity)) = active_notes.remove(&key) {
            let duration = tick.saturating_sub(start_tick).max(config.min_note_duration);
            
            note_events.push(MidiNoteEvent {
                channel,
                note: midi_note,
                velocity: prev_velocity,
                start_tick,
                duration_ticks: duration,
            });
        }
        
        // Start the new note
        active_notes.insert(key, (tick, velocity));
    }
    
    // End any remaining active notes
    if let Some(last_event) = audio_events.last() {
        let end_tick = seconds_to_ticks(last_event.timestamp + 1.0, config.ticks_per_quarter);
        
        for ((channel, note), (start_tick, velocity)) in active_notes.drain() {
            let duration = end_tick.saturating_sub(start_tick).max(config.min_note_duration);
            
            note_events.push(MidiNoteEvent {
                channel,
                note,
                velocity,
                start_tick,
                duration_ticks: duration,
            });
        }
    }
    
    // Sort events by time
    note_events.sort_by_key(|event| event.start_tick);
    cc_events.sort_by_key(|event| event.tick);
    
    Ok(MidiEventCollection {
        note_events,
        cc_events,
    })
}

/// Generate a MIDI file from MIDI note events
pub fn generate_midi_file(
    midi_events: Vec<MidiNoteEvent>,
    config: &ConversionConfig,
) -> Result<Vec<u8>> {
    let collection = MidiEventCollection {
        note_events: midi_events,
        cc_events: Vec::new(),
    };
    generate_midi_file_with_cc(collection, config)
}

/// Generate a MIDI file from MIDI events including CC data
pub fn generate_midi_file_with_cc(
    event_collection: MidiEventCollection,
    config: &ConversionConfig,
) -> Result<Vec<u8>> {
    // Create MIDI header
    let timing = Timing::Metrical(config.ticks_per_quarter.try_into().unwrap_or(480.into()));
    let header = Header::new(Format::Parallel, timing);
    
    // Group events by channel
    let mut channels: HashMap<u8, (Vec<&MidiNoteEvent>, Vec<&MidiCCEvent>)> = HashMap::new();
    
    for event in &event_collection.note_events {
        channels.entry(event.channel).or_insert_with(|| (Vec::new(), Vec::new())).0.push(event);
    }
    
    for event in &event_collection.cc_events {
        channels.entry(event.channel).or_insert_with(|| (Vec::new(), Vec::new())).1.push(event);
    }
    
    let mut tracks = Vec::new();
    
    // Create a track for each channel
    for (channel, (note_events, cc_events)) in channels {
        let mut track_events = Vec::new();
        let mut current_tick = 0u32;
        
        // Create a combined list of all events with their timestamps
        let mut all_events: Vec<(u32, String)> = Vec::new();
        
        // Add note events
        for event in &note_events {
            all_events.push((event.start_tick, format!("note_on:{}:{}:{}", event.note, event.velocity, event.duration_ticks)));
        }
        
        // Add CC events
        for event in &cc_events {
            all_events.push((event.tick, format!("cc:{}:{}", event.controller, event.value)));
        }
        
        // Sort all events by timestamp
        all_events.sort_by_key(|event| event.0);
        
        // Process events in chronological order
        let mut active_notes: HashMap<u8, u32> = HashMap::new(); // note -> note_off_tick
        
        for (tick, event_data) in all_events {
            let _delta_time = tick.saturating_sub(current_tick);
            
            // Process any note-off events that should occur before or at this time
            let mut notes_to_turn_off: Vec<(u8, u32)> = active_notes.iter()
                .filter(|(_, &note_off_tick)| note_off_tick <= tick)
                .map(|(&note, &note_off_tick)| (note, note_off_tick))
                .collect();
            notes_to_turn_off.sort_by_key(|(_, note_off_tick)| *note_off_tick);
            
            for (note, note_off_tick) in notes_to_turn_off {
                let note_off_delta = note_off_tick.saturating_sub(current_tick);
                
                track_events.push(TrackEvent {
                    delta: note_off_delta.try_into().unwrap_or(0.into()),
                    kind: TrackEventKind::Midi {
                        channel: channel.into(),
                        message: MidiMessage::NoteOff {
                            key: note.into(),
                            vel: 0.into(),
                        },
                    },
                });
                
                current_tick = note_off_tick;
                active_notes.remove(&note);
            }
            
            // Process the current event
            let event_delta = tick.saturating_sub(current_tick);
            
            if event_data.starts_with("note_on:") {
                let parts: Vec<&str> = event_data.split(':').collect();
                if parts.len() >= 4 {
                    let note: u8 = parts[1].parse().unwrap_or(60);
                    let velocity: u8 = parts[2].parse().unwrap_or(64);
                    let duration: u32 = parts[3].parse().unwrap_or(100);
                    
                    track_events.push(TrackEvent {
                        delta: event_delta.try_into().unwrap_or(0.into()),
                        kind: TrackEventKind::Midi {
                            channel: channel.into(),
                            message: MidiMessage::NoteOn {
                                key: note.into(),
                                vel: velocity.into(),
                            },
                        },
                    });
                    
                    // Schedule note off
                    active_notes.insert(note, tick + duration);
                }
            } else if event_data.starts_with("cc:") {
                let parts: Vec<&str> = event_data.split(':').collect();
                if parts.len() >= 3 {
                    let controller: u8 = parts[1].parse().unwrap_or(1);
                    let value: u8 = parts[2].parse().unwrap_or(0);
                    
                    track_events.push(TrackEvent {
                        delta: event_delta.try_into().unwrap_or(0.into()),
                        kind: TrackEventKind::Midi {
                            channel: channel.into(),
                            message: MidiMessage::Controller {
                                controller: controller.into(),
                                value: value.into(),
                            },
                        },
                    });
                }
            }
            
            current_tick = tick;
        }
        
        // Turn off any remaining active notes
        let mut remaining_notes: Vec<(u8, u32)> = active_notes.into_iter().collect();
        remaining_notes.sort_by_key(|(_, note_off_tick)| *note_off_tick);
        
        for (note, note_off_tick) in remaining_notes {
            let note_off_delta = note_off_tick.saturating_sub(current_tick);
            
            track_events.push(TrackEvent {
                delta: note_off_delta.try_into().unwrap_or(0.into()),
                kind: TrackEventKind::Midi {
                    channel: channel.into(),
                    message: MidiMessage::NoteOff {
                        key: note.into(),
                        vel: 0.into(),
                    },
                },
            });
            
            current_tick = note_off_tick;
        }
        
        // Add end of track event
        track_events.push(TrackEvent {
            delta: 0.into(),
            kind: TrackEventKind::Meta(midly::MetaMessage::EndOfTrack),
        });
        
        tracks.push(Track::from(track_events));
    }
    
    // If no tracks were created, create an empty track
    if tracks.is_empty() {
        let track = Track::from(vec![TrackEvent {
            delta: 0.into(),
            kind: TrackEventKind::Meta(midly::MetaMessage::EndOfTrack),
        }]);
        tracks.push(track);
    }
    
    // Create the MIDI file
    let smf = midly::Smf {
        header,
        tracks,
    };
    
    // Write to bytes
    let mut buffer = Vec::new();
    smf.write(&mut buffer)
        .map_err(|e| crate::ConversionError::MidiWriteError(e.to_string()))?;
    
    Ok(buffer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::AudioEvent;
    
    #[test]
    fn test_frequency_to_midi_note() {
        // A4 = 440Hz = MIDI note 69
        assert_eq!(frequency_to_midi_note(440.0), 69);
        
        // C4 = ~261.63Hz = MIDI note 60
        assert_eq!(frequency_to_midi_note(261.63), 60);
        
        // Silence
        assert_eq!(frequency_to_midi_note(0.0), 0);
        
        // Very high frequency (should clamp to 127)
        assert_eq!(frequency_to_midi_note(20000.0), 127);
    }
    
    #[test]
    fn test_amplitude_to_velocity() {
        assert_eq!(amplitude_to_velocity(1.0, 64), 127);
        assert_eq!(amplitude_to_velocity(0.5, 64), 64);
        assert_eq!(amplitude_to_velocity(0.0, 64), 0);
        
        // Should respect minimum velocity
        let vel = amplitude_to_velocity(0.1, 64);
        assert!(vel >= 16); // default_velocity / 4
    }
    
    #[test]
    fn test_seconds_to_ticks() {
        // At 120 BPM with 480 ticks per quarter note
        // 0.5 seconds = 1 quarter note = 480 ticks
        assert_eq!(seconds_to_ticks(0.5, 480), 480);
        assert_eq!(seconds_to_ticks(1.0, 480), 960);
        assert_eq!(seconds_to_ticks(0.0, 480), 0);
    }
    
    #[test]
    fn test_convert_to_midi_events() {
        let events = vec![
            AudioEvent {
                line_number: 1,
                timestamp: 0.0,
                frequency: 440.0,
                amplitude: 0.5,
                channel: Some(1),
            },
            AudioEvent {
                line_number: 2,
                timestamp: 1.0,
                frequency: 0.0,
                amplitude: 0.0,
                channel: None,
            },
        ];
        
        let config = ConversionConfig::default();
        let midi_events = convert_to_midi_events(&events, &config).unwrap();
        
        assert_eq!(midi_events.len(), 1);
        assert_eq!(midi_events[0].note, 69); // A4
        assert_eq!(midi_events[0].channel, 0); // 0-based channel
    }
}
