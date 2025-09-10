use clap::{Arg, Command};
use csv_to_midi_core::{convert_csv_to_midi, ConversionConfig};
use std::fs;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let matches = Command::new("csv-to-midi")
        .version("0.1.0")
        .author("Your Name <your.email@example.com>")
        .about("Convert CSV audio analysis data to MIDI files")
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .value_name("FILE")
                .help("Input CSV file path")
                .required(true),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("FILE")
                .help("Output MIDI file path")
                .required(true),
        )
        .arg(
            Arg::new("ticks")
                .short('t')
                .long("ticks-per-quarter")
                .value_name("NUMBER")
                .help("MIDI ticks per quarter note (default: 480)")
                .default_value("480"),
        )
        .arg(
            Arg::new("velocity")
                .short('v')
                .long("default-velocity")
                .value_name("NUMBER")
                .help("Default MIDI velocity (0-127, default: 64)")
                .default_value("64"),
        )
        .arg(
            Arg::new("duration")
                .short('d')
                .long("min-duration")
                .value_name("NUMBER")
                .help("Minimum note duration in ticks (default: 100)")
                .default_value("100"),
        )
        .get_matches();

    // Get command line arguments
    let input_path = matches.get_one::<String>("input").unwrap();
    let output_path = matches.get_one::<String>("output").unwrap();
    
    let ticks_per_quarter: u16 = matches
        .get_one::<String>("ticks")
        .unwrap()
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid ticks per quarter note"))?;
    
    let default_velocity: u8 = matches
        .get_one::<String>("velocity")
        .unwrap()
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid default velocity"))?;
    
    let min_note_duration: u32 = matches
        .get_one::<String>("duration")
        .unwrap()
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid minimum duration"))?;

    // Validate input file exists
    if !Path::new(input_path).exists() {
        return Err(anyhow::anyhow!("Input file does not exist: {}", input_path));
    }

    // Validate velocity range
    if default_velocity > 127 {
        return Err(anyhow::anyhow!("Default velocity must be between 0 and 127"));
    }

    println!("Converting CSV to MIDI...");
    println!("Input: {}", input_path);
    println!("Output: {}", output_path);
    println!("Ticks per quarter: {}", ticks_per_quarter);
    println!("Default velocity: {}", default_velocity);
    println!("Minimum duration: {}", min_note_duration);

    // Read the input CSV file
    let csv_data = fs::read(input_path)
        .map_err(|e| anyhow::anyhow!("Failed to read input file: {}", e))?;

    // Create conversion configuration
    let config = ConversionConfig {
        ticks_per_quarter,
        min_note_duration,
        default_velocity,
        base_octave: 4,
    };

    // Convert CSV to MIDI
    let midi_data = convert_csv_to_midi(csv_data.as_slice(), config)
        .map_err(|e| anyhow::anyhow!("Conversion failed: {}", e))?;

    // Write the output MIDI file
    fs::write(output_path, midi_data)
        .map_err(|e| anyhow::anyhow!("Failed to write output file: {}", e))?;

    println!("Conversion completed successfully!");
    println!("Generated MIDI file: {} ({} bytes)", output_path, fs::metadata(output_path)?.len());

    Ok(())
}
