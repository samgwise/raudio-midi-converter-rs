//! MIDI post-processing module
//!
//! This module provides functionality to clean up and simplify MIDI output
//! through various post-processing techniques including pitch filtering,
//! velocity expansion, note joining, and other simplification operations.

use crate::{Result, ConversionError};
use crate::midi::{MidiNoteEvent, MidiCCEvent, MidiEventCollection};
use std::collections::HashMap;

/// Configuration for MIDI post-processing operations
#[derive(Debug, Clone)]
pub struct PostProcessingConfig {
    /// Enable pitch range filtering
    pub enable_pitch_filtering: bool,
    /// Minimum MIDI note number (0-127)
    pub min_midi_note: u8,
    /// Maximum MIDI note number (0-127)  
    pub max_midi_note: u8,
    
    /// Enable velocity expansion for quiet notes
    pub enable_velocity_expansion: bool,
    /// Velocity threshold below which to apply expansion (0-127)
    pub velocity_threshold: u8,
    /// Velocity expansion factor (1.0 = no change, >1.0 = louder)
    pub velocity_expansion_factor: f32,
    /// Maximum velocity after expansion (0-127)
    pub max_expanded_velocity: u8,
    
    /// Enable note joining and simplification
    pub enable_note_joining: bool,
    /// Minimum note duration in ticks (notes shorter than this may be joined or removed)
    pub min_note_duration: u32,
    /// Maximum gap between notes to join (in ticks)
    pub max_join_gap: u32,
    /// Remove notes shorter than this duration entirely (in ticks)
    pub remove_short_notes_threshold: u32,
    
    /// Enable duplicate note removal
    pub enable_duplicate_removal: bool,
    /// Time window for duplicate detection (in ticks)
    pub duplicate_time_window: u32,
    
    /// Enable CC event simplification
    pub enable_cc_simplification: bool,
    /// Minimum change in CC value to keep event (0-127)
    pub cc_min_change_threshold: u8,
    /// Maximum number of CC events per second per controller
    pub cc_max_events_per_second: f32,
}

impl Default for PostProcessingConfig {
    fn default() -> Self {
        Self {
            enable_pitch_filtering: false,
            min_midi_note: 21,  // A0
            max_midi_note: 108, // C8
            
            enable_velocity_expansion: false,
            velocity_threshold: 40,
            velocity_expansion_factor: 1.5,
            max_expanded_velocity: 100,
            
            enable_note_joining: true,
            min_note_duration: 50,      // 50 ticks minimum
            max_join_gap: 24,           // 24 ticks max gap (1/20th of quarter note at 480 tpq)
            remove_short_notes_threshold: 12, // Remove notes shorter than 12 ticks
            
            enable_duplicate_removal: true,
            duplicate_time_window: 12,  // 12 ticks window
            
            enable_cc_simplification: true,
            cc_min_change_threshold: 2, // Minimum change of 2 CC values
            cc_max_events_per_second: 20.0, // Max 20 CC events per second
        }
    }
}

/// Apply post-processing to a MIDI event collection
pub fn post_process_midi(
    mut collection: MidiEventCollection,
    config: &PostProcessingConfig,
    ticks_per_quarter: u16,
) -> Result<MidiEventCollection> {
    // Apply pitch filtering first
    if config.enable_pitch_filtering {
        collection.note_events = filter_pitch_range(collection.note_events, config)?;
    }
    
    // Apply velocity expansion
    if config.enable_velocity_expansion {
        collection.note_events = expand_low_velocities(collection.note_events, config)?;
    }
    
    // Remove very short notes if threshold is set
    if config.enable_note_joining && config.remove_short_notes_threshold > 0 {
        collection.note_events = remove_short_notes(collection.note_events, config)?;
    }
    
    // Join overlapping/adjacent notes
    if config.enable_note_joining {
        collection.note_events = join_overlapping_notes(collection.note_events, config)?;
    }
    
    // Remove duplicate notes
    if config.enable_duplicate_removal {
        collection.note_events = remove_duplicate_notes(collection.note_events, config)?;
    }
    
    // Simplify CC events
    if config.enable_cc_simplification {
        collection.cc_events = simplify_cc_events(collection.cc_events, config, ticks_per_quarter)?;
    }
    
    // Sort events by time to ensure proper ordering
    collection.note_events.sort_by_key(|event| event.start_tick);
    collection.cc_events.sort_by_key(|event| event.tick);
    
    Ok(collection)
}

/// Filter MIDI notes by pitch range
fn filter_pitch_range(
    note_events: Vec<MidiNoteEvent>,
    config: &PostProcessingConfig,
) -> Result<Vec<MidiNoteEvent>> {
    let filtered = note_events
        .into_iter()
        .filter(|event| {
            event.note >= config.min_midi_note && event.note <= config.max_midi_note
        })
        .collect();
    
    Ok(filtered)
}

/// Expand velocities of quiet notes
fn expand_low_velocities(
    mut note_events: Vec<MidiNoteEvent>,
    config: &PostProcessingConfig,
) -> Result<Vec<MidiNoteEvent>> {
    for event in &mut note_events {
        if event.velocity <= config.velocity_threshold {
            let expanded = (event.velocity as f32 * config.velocity_expansion_factor).round() as u8;
            event.velocity = expanded.min(config.max_expanded_velocity).max(1);
        }
    }
    
    Ok(note_events)
}

/// Remove notes shorter than threshold
fn remove_short_notes(
    note_events: Vec<MidiNoteEvent>,
    config: &PostProcessingConfig,
) -> Result<Vec<MidiNoteEvent>> {
    let filtered = note_events
        .into_iter()
        .filter(|event| event.duration_ticks >= config.remove_short_notes_threshold)
        .collect();
    
    Ok(filtered)
}

/// Join overlapping or adjacent notes of the same pitch and channel
fn join_overlapping_notes(
    mut note_events: Vec<MidiNoteEvent>,
    config: &PostProcessingConfig,
) -> Result<Vec<MidiNoteEvent>> {
    if note_events.is_empty() {
        return Ok(note_events);
    }
    
    // Sort by channel, note, then start time
    note_events.sort_by(|a, b| {
        a.channel.cmp(&b.channel)
            .then(a.note.cmp(&b.note))
            .then(a.start_tick.cmp(&b.start_tick))
    });
    
    let mut joined_events = Vec::new();
    let mut current_event: Option<MidiNoteEvent> = None;
    
    for event in note_events {
        match &mut current_event {
            None => {
                current_event = Some(event);
            }
            Some(ref mut current) => {
                // Check if this event can be joined with the current one
                if current.channel == event.channel
                    && current.note == event.note
                    && event.start_tick <= (current.start_tick + current.duration_ticks + config.max_join_gap)
                {
                    // Join the notes by extending the duration
                    let current_end = current.start_tick + current.duration_ticks;
                    let event_end = event.start_tick + event.duration_ticks;
                    let new_end = current_end.max(event_end);
                    
                    current.duration_ticks = new_end - current.start_tick;
                    // Use the higher velocity
                    current.velocity = current.velocity.max(event.velocity);
                } else {
                    // Can't join, save the current event and start a new one
                    joined_events.push(current.clone());
                    *current = event;
                }
            }
        }
    }
    
    // Don't forget the last event
    if let Some(event) = current_event {
        joined_events.push(event);
    }
    
    Ok(joined_events)
}

/// Remove duplicate notes (same channel, pitch, and time)
fn remove_duplicate_notes(
    mut note_events: Vec<MidiNoteEvent>,
    config: &PostProcessingConfig,
) -> Result<Vec<MidiNoteEvent>> {
    if note_events.is_empty() {
        return Ok(note_events);
    }
    
    note_events.sort_by_key(|event| (event.channel, event.note, event.start_tick));
    
    let mut deduplicated = Vec::new();
    let mut last_event: Option<&MidiNoteEvent> = None;
    
    for event in &note_events {
        let should_keep = match last_event {
            None => true,
            Some(last) => {
                // Consider duplicate if same channel, note, and within time window
                !(last.channel == event.channel
                    && last.note == event.note
                    && event.start_tick.saturating_sub(last.start_tick) <= config.duplicate_time_window)
            }
        };
        
        if should_keep {
            deduplicated.push(event.clone());
            last_event = Some(event);
        }
    }
    
    Ok(deduplicated)
}

/// Simplify CC events by removing redundant data
fn simplify_cc_events(
    mut cc_events: Vec<MidiCCEvent>,
    config: &PostProcessingConfig,
    ticks_per_quarter: u16,
) -> Result<Vec<MidiCCEvent>> {
    if cc_events.is_empty() {
        return Ok(cc_events);
    }
    
    // Sort by channel, controller, then time
    cc_events.sort_by(|a, b| {
        a.channel.cmp(&b.channel)
            .then(a.controller.cmp(&b.controller))
            .then(a.tick.cmp(&b.tick))
    });
    
    let mut simplified = Vec::new();
    let mut last_values: HashMap<(u8, u8), (u8, u32)> = HashMap::new(); // (channel, controller) -> (value, tick)
    
    let ticks_per_second = ticks_per_quarter as f32 * 2.0; // Assuming 120 BPM (2 beats per second)
    let min_tick_interval = (ticks_per_second / config.cc_max_events_per_second) as u32;
    
    for event in cc_events {
        let key = (event.channel, event.controller);
        let should_keep = match last_values.get(&key) {
            None => true,
            Some(&(last_value, last_tick)) => {
                let value_change = if event.value > last_value {
                    event.value - last_value
                } else {
                    last_value - event.value
                };
                
                let time_elapsed = event.tick.saturating_sub(last_tick);
                
                // Keep if value change is significant or enough time has passed
                value_change >= config.cc_min_change_threshold || time_elapsed >= min_tick_interval
            }
        };
        
        if should_keep {
            simplified.push(event.clone());
            last_values.insert(key, (event.value, event.tick));
        }
    }
    
    Ok(simplified)
}

/// Statistics about post-processing operations
#[derive(Debug, Clone)]
pub struct PostProcessingStats {
    pub original_note_count: usize,
    pub final_note_count: usize,
    pub notes_removed_by_pitch_filter: usize,
    pub notes_removed_as_too_short: usize,
    pub notes_joined: usize,
    pub duplicates_removed: usize,
    
    pub original_cc_count: usize,
    pub final_cc_count: usize,
    pub cc_events_simplified: usize,
}

/// Apply post-processing with detailed statistics
pub fn post_process_midi_with_stats(
    collection: MidiEventCollection,
    config: &PostProcessingConfig,
    ticks_per_quarter: u16,
) -> Result<(MidiEventCollection, PostProcessingStats)> {
    let original_note_count = collection.note_events.len();
    let original_cc_count = collection.cc_events.len();
    
    let processed = post_process_midi(collection, config, ticks_per_quarter)?;
    
    let stats = PostProcessingStats {
        original_note_count,
        final_note_count: processed.note_events.len(),
        notes_removed_by_pitch_filter: 0, // TODO: Could track these individually
        notes_removed_as_too_short: 0,
        notes_joined: 0,
        duplicates_removed: 0,
        
        original_cc_count,
        final_cc_count: processed.cc_events.len(),
        cc_events_simplified: original_cc_count.saturating_sub(processed.cc_events.len()),
    };
    
    Ok((processed, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::midi::{MidiNoteEvent, MidiCCEvent};

    #[test]
    fn test_pitch_filtering() {
        let events = vec![
            MidiNoteEvent { channel: 0, note: 20, velocity: 64, start_tick: 0, duration_ticks: 100 }, // Too low
            MidiNoteEvent { channel: 0, note: 60, velocity: 64, start_tick: 0, duration_ticks: 100 }, // OK
            MidiNoteEvent { channel: 0, note: 120, velocity: 64, start_tick: 0, duration_ticks: 100 }, // Too high
        ];
        
        let config = PostProcessingConfig {
            enable_pitch_filtering: true,
            min_midi_note: 21,
            max_midi_note: 108,
            ..Default::default()
        };
        
        let filtered = filter_pitch_range(events, &config).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].note, 60);
    }

    #[test]
    fn test_velocity_expansion() {
        let events = vec![
            MidiNoteEvent { channel: 0, note: 60, velocity: 30, start_tick: 0, duration_ticks: 100 }, // Below threshold
            MidiNoteEvent { channel: 0, note: 61, velocity: 80, start_tick: 0, duration_ticks: 100 }, // Above threshold
        ];
        
        let config = PostProcessingConfig {
            enable_velocity_expansion: true,
            velocity_threshold: 40,
            velocity_expansion_factor: 2.0,
            max_expanded_velocity: 100,
            ..Default::default()
        };
        
        let expanded = expand_low_velocities(events, &config).unwrap();
        assert_eq!(expanded[0].velocity, 60); // 30 * 2.0
        assert_eq!(expanded[1].velocity, 80); // Unchanged
    }

    #[test]
    fn test_note_joining() {
        let events = vec![
            MidiNoteEvent { channel: 0, note: 60, velocity: 64, start_tick: 0, duration_ticks: 100 },
            MidiNoteEvent { channel: 0, note: 60, velocity: 70, start_tick: 90, duration_ticks: 100 }, // Overlaps
        ];
        
        let config = PostProcessingConfig {
            enable_note_joining: true,
            max_join_gap: 50,
            ..Default::default()
        };
        
        let joined = join_overlapping_notes(events, &config).unwrap();
        assert_eq!(joined.len(), 1);
        assert_eq!(joined[0].duration_ticks, 190); // Extended to cover both notes
        assert_eq!(joined[0].velocity, 70); // Higher velocity
    }

    #[test]
    fn test_short_note_removal() {
        let events = vec![
            MidiNoteEvent { channel: 0, note: 60, velocity: 64, start_tick: 0, duration_ticks: 5 }, // Too short
            MidiNoteEvent { channel: 0, note: 61, velocity: 64, start_tick: 0, duration_ticks: 50 }, // OK
        ];
        
        let config = PostProcessingConfig {
            enable_note_joining: true,
            remove_short_notes_threshold: 10,
            ..Default::default()
        };
        
        let filtered = remove_short_notes(events, &config).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].note, 61);
    }
}
