use csv_to_midi_core::{convert_csv_string_to_midi, ConversionConfig};
use wasm_bindgen::prelude::*;

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// Configuration for CSV to MIDI conversion (WASM-exposed version)
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct WasmConversionConfig {
    ticks_per_quarter: u16,
    min_note_duration: u32,
    default_velocity: u8,
    base_octave: u8,
}

#[wasm_bindgen]
impl WasmConversionConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(
        ticks_per_quarter: Option<u16>,
        min_note_duration: Option<u32>,
        default_velocity: Option<u8>,
        base_octave: Option<u8>,
    ) -> WasmConversionConfig {
        WasmConversionConfig {
            ticks_per_quarter: ticks_per_quarter.unwrap_or(480),
            min_note_duration: min_note_duration.unwrap_or(100),
            default_velocity: default_velocity.unwrap_or(64),
            base_octave: base_octave.unwrap_or(4),
        }
    }
    
    #[wasm_bindgen(getter)]
    pub fn ticks_per_quarter(&self) -> u16 {
        self.ticks_per_quarter
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_ticks_per_quarter(&mut self, value: u16) {
        self.ticks_per_quarter = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn min_note_duration(&self) -> u32 {
        self.min_note_duration
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_min_note_duration(&mut self, value: u32) {
        self.min_note_duration = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn default_velocity(&self) -> u8 {
        self.default_velocity
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_default_velocity(&mut self, value: u8) {
        self.default_velocity = value;
    }
    
    #[wasm_bindgen(getter)]
    pub fn base_octave(&self) -> u8 {
        self.base_octave
    }
    
    #[wasm_bindgen(setter)]
    pub fn set_base_octave(&mut self, value: u8) {
        self.base_octave = value;
    }
}

impl From<WasmConversionConfig> for ConversionConfig {
    fn from(config: WasmConversionConfig) -> Self {
        ConversionConfig {
            ticks_per_quarter: config.ticks_per_quarter,
            min_note_duration: config.min_note_duration,
            default_velocity: config.default_velocity,
            base_octave: config.base_octave,
        }
    }
}

/// Convert CSV string to MIDI byte array
/// 
/// # Arguments
/// * `csv_data` - CSV string containing audio analysis data
/// * `config` - Configuration object for conversion options
/// 
/// # Returns
/// * Uint8Array containing MIDI file data, or throws error if conversion fails
#[wasm_bindgen]
pub fn convert_csv_to_midi_wasm(
    csv_data: &str,
    config: &WasmConversionConfig,
) -> Result<js_sys::Uint8Array, JsValue> {
    console_log!("Starting CSV to MIDI conversion...");
    console_log!("CSV data length: {} bytes", csv_data.len());
    
    let core_config: ConversionConfig = config.clone().into();
    
    let midi_data = convert_csv_string_to_midi(csv_data, core_config)
        .map_err(|e| JsValue::from_str(&format!("Conversion error: {}", e)))?;
    
    console_log!("Conversion successful! Generated {} bytes of MIDI data", midi_data.len());
    
    Ok(js_sys::Uint8Array::from(&midi_data[..]))
}

/// Create a default configuration
#[wasm_bindgen]
pub fn create_default_config() -> WasmConversionConfig {
    WasmConversionConfig::new(None, None, None, None)
}

/// Get library version information
#[wasm_bindgen]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
