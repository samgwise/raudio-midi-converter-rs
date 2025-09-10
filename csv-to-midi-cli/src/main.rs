use clap::{Arg, Command};
use csv_to_midi_core::{convert_csv_to_midi, ConversionConfig};

#[cfg(feature = "audio")]
use csv_to_midi_core::{convert_audio_to_midi, AudioAnalysisConfig, CCMappingConfig};
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
                .help({
                    #[cfg(feature = "audio")]
                    {
                        "Input file path (CSV or audio file: .wav, .flac)"
                    }
                    #[cfg(not(feature = "audio"))]
                    {
                        "Input CSV file path"
                    }
                })
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
        );
    
    // Add audio analysis arguments conditionally
    #[cfg(feature = "audio")]
    let matches = {
        matches
            .arg(
                Arg::new("fmin")
                    .long("fmin")
                    .value_name("HZ")
                    .help("Minimum frequency for audio analysis in Hz (default: 65.0)")
                    .default_value("65.0"),
            )
            .arg(
                Arg::new("fmax")
                    .long("fmax")
                    .value_name("HZ")
                    .help("Maximum frequency for audio analysis in Hz (default: 2093.0)")
                    .default_value("2093.0"),
            )
            .arg(
                Arg::new("threshold")
                    .long("threshold")
                    .value_name("NUMBER")
                    .help("Threshold for voiced/unvoiced detection (0.0-1.0, default: 0.1)")
                    .default_value("0.1"),
            )
            .arg(
                Arg::new("frame-size")
                    .long("frame-size")
                    .value_name("SAMPLES")
                    .help("Frame size for audio analysis in samples (default: 2048)")
                    .default_value("2048"),
            )
            .arg(
                Arg::new("hop-size")
                    .long("hop-size")
                    .value_name("SAMPLES")
                    .help("Hop size for audio analysis in samples (default: 512)")
                    .default_value("512"),
            )
    };
    
    #[cfg(not(feature = "audio"))]
    let matches = matches;
    
    let matches = matches
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

    // Parse audio-specific arguments (only when audio feature is enabled)
    #[cfg(feature = "audio")]
    let (fmin, fmax, threshold, frame_size, hop_size) = {
        let fmin: f64 = matches
            .get_one::<String>("fmin")
            .unwrap()
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid minimum frequency"))?;
        
        let fmax: f64 = matches
            .get_one::<String>("fmax")
            .unwrap()
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid maximum frequency"))?;
        
        let threshold: f64 = matches
            .get_one::<String>("threshold")
            .unwrap()
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid threshold"))?;
        
        let frame_size: usize = matches
            .get_one::<String>("frame-size")
            .unwrap()
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid frame size"))?;
        
        let hop_size: usize = matches
            .get_one::<String>("hop-size")
            .unwrap()
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid hop size"))?;
        
        (fmin, fmax, threshold, frame_size, hop_size)
    };

    // Validate input file exists
    if !Path::new(input_path).exists() {
        return Err(anyhow::anyhow!("Input file does not exist: {}", input_path));
    }

    // Validate parameters
    if default_velocity > 127 {
        return Err(anyhow::anyhow!("Default velocity must be between 0 and 127"));
    }
    
    // Audio-specific validation (only when audio feature is enabled)
    #[cfg(feature = "audio")]
    {
        if threshold < 0.0 || threshold > 1.0 {
            return Err(anyhow::anyhow!("Threshold must be between 0.0 and 1.0"));
        }
        
        if fmin >= fmax {
            return Err(anyhow::anyhow!("Minimum frequency must be less than maximum frequency"));
        }
    }

    // Determine file type from extension
    let input_path_obj = Path::new(input_path);
    let is_audio_file = {
        #[cfg(feature = "audio")]
        {
            match input_path_obj.extension().and_then(|ext| ext.to_str()) {
                Some("wav") | Some("WAV") | Some("flac") | Some("FLAC") => true,
                Some("csv") | Some("CSV") => false,
                _ => {
                    // Try to detect based on content or assume CSV
                    false
                }
            }
        }
        #[cfg(not(feature = "audio"))]
        {
            // Without audio feature, check if user tried to use audio file
            match input_path_obj.extension().and_then(|ext| ext.to_str()) {
                Some("wav") | Some("WAV") | Some("flac") | Some("FLAC") => {
                    return Err(anyhow::anyhow!(
                        "Audio file processing is not available. Build with --features audio to enable audio support."
                    ));
                },
                _ => false, // Assume CSV
            }
        }
    };

    // Create MIDI conversion configuration
    let midi_config = ConversionConfig {
        ticks_per_quarter,
        min_note_duration,
        default_velocity,
        base_octave: 4,
    };

    let midi_data = if is_audio_file {
        #[cfg(feature = "audio")]
        {
            // Audio file processing
            println!("Analyzing audio file with pitch detection...");
            println!("Input: {} (audio)", input_path);
            println!("Output: {}", output_path);
            println!("Sample rate: 44100 Hz");
            println!("Frame size: {} samples", frame_size);
            println!("Hop size: {} samples", hop_size);
            println!("Frequency range: {:.1} - {:.1} Hz", fmin, fmax);
            println!("Threshold: {:.2}", threshold);
            println!("MIDI settings: {} ticks/quarter, velocity {}, min duration {}", 
                     ticks_per_quarter, default_velocity, min_note_duration);

            // Create audio analysis configuration with default CC mapping and enhanced analysis enabled
            let audio_config = AudioAnalysisConfig {
                target_sample_rate: 44100,
                frame_size,
                hop_size,
                fmin,
                fmax,
                threshold,
                cc_mapping: CCMappingConfig::default(),
                enable_spectral_analysis: true,
                enable_harmonicity_analysis: true,
                enable_zero_crossing_analysis: true,
                enable_peak_normalization: true,
                normalization_target: 0.95,
            };

            // Convert audio to MIDI
            convert_audio_to_midi(input_path, &audio_config, &midi_config)
                .map_err(|e| anyhow::anyhow!("Audio analysis failed: {}", e))?
        }
        #[cfg(not(feature = "audio"))]
        {
            // This should never be reached due to earlier checks, but just in case
            return Err(anyhow::anyhow!(
                "Audio file detected but audio feature is not enabled. Build with --features audio."
            ));
        }
    } else {
        // CSV file processing (existing logic)
        println!("Converting CSV to MIDI...");
        println!("Input: {} (CSV)", input_path);
        println!("Output: {}", output_path);
        println!("Ticks per quarter: {}", ticks_per_quarter);
        println!("Default velocity: {}", default_velocity);
        println!("Minimum duration: {}", min_note_duration);

        // Read the input CSV file
        let csv_data = fs::read(input_path)
            .map_err(|e| anyhow::anyhow!("Failed to read input file: {}", e))?;

        // Convert CSV to MIDI
        convert_csv_to_midi(csv_data.as_slice(), midi_config)
            .map_err(|e| anyhow::anyhow!("CSV conversion failed: {}", e))?
    };

    // Write the output MIDI file
    fs::write(output_path, &midi_data)
        .map_err(|e| anyhow::anyhow!("Failed to write output file: {}", e))?;

    println!("Conversion completed successfully!");
    println!("Generated MIDI file: {} ({} bytes)", output_path, midi_data.len());

    Ok(())
}
