//! ferromode_run — minimal CLI: reads signal CSV, runs ferromode EMD, writes IMFs CSV.
//!
//! Usage:
//!   ferromode_run --input signal.csv --output imfs.csv [--max-imfs 12]
//!
//! Input CSV:  one column, one sample per line, no header.
//! Output CSV: header row "residue,imf1,imf2,...", then one row per sample.

use std::fs;
use std::io::{self, BufRead, Write};

use ferromode::algorithms::emd::{emd, EmdConfig};
use ferromode::boundary::BoundaryConditionType;
use ferromode::sifting::{SiftingConfig, SplineEndCondition};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── arg parsing (no dep, hand-rolled) ────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        args.windows(2)
            .find(|w| w[0] == flag)
            .map(|w| w[1].clone())
    };
    let has = |flag: &str| -> bool { args.iter().any(|a| a == flag) };

    let input_path  = get("--input").unwrap_or_else(|| { eprintln!("--input required"); std::process::exit(1) });
    let output_path = get("--output").unwrap_or_else(|| { eprintln!("--output required"); std::process::exit(1) });
    let max_imfs: usize  = get("--max-imfs").and_then(|s| s.parse().ok()).unwrap_or(0);
    let sd_thr: f64      = get("--sd-thr").and_then(|s| s.parse().ok()).unwrap_or(0.2);
    let s_number: usize  = get("--s-number").and_then(|s| s.parse().ok()).unwrap_or(5);
    let max_sift: usize  = get("--max-sift").and_then(|s| s.parse().ok()).unwrap_or(100);
    // --sd-only: match PyEMD behaviour — stop on SD threshold alone, no S-number
    let sd_only = has("--sd-only");

    let boundary = match get("--boundary").as_deref() {
        Some("extrema") | None      => BoundaryConditionType::ExtremasMirror,  // default, matches PyEMD
        Some("mirror")              => BoundaryConditionType::MirrorEven,
        Some("periodic")            => BoundaryConditionType::Periodic,
        Some("slope")               => BoundaryConditionType::Slope,
        Some("ar")                  => BoundaryConditionType::ARModel,
        Some("waveform")            => BoundaryConditionType::WaveformMatching,
        Some(other) => { eprintln!("Unknown boundary: {other}"); std::process::exit(1) }
    };

    // ── load signal ───────────────────────────────────────────────────────────
    let file   = fs::File::open(&input_path)?;
    let signal: Vec<f64> = io::BufReader::new(file)
        .lines()
        .filter_map(|l| l.ok())
        .filter_map(|l| {
            let t = l.trim().to_string();
            if t.is_empty() || t.starts_with('#') { None } else { t.parse::<f64>().ok() }
        })
        .collect();

    eprintln!("Loaded {} samples from {}", signal.len(), input_path);
    eprintln!("Config: sd_thr={sd_thr}  s_number={}  max_sift={max_sift}  boundary={boundary:?}  sd_only={sd_only}",
              if sd_only { 0 } else { s_number });

    // ── run EMD ───────────────────────────────────────────────────────────────
    let spline_ec = match get("--spline").as_deref() {
        Some("natural")   => SplineEndCondition::Natural,
        Some("periodic")  => SplineEndCondition::Periodic,
        _                 => SplineEndCondition::NotAKnot,  // default, matches scipy/PyEMD
    };

    let sifting_config = SiftingConfig {
        max_sifting_iterations: max_sift,
        sd_threshold: sd_thr,
        s_number: if sd_only { 0 } else { s_number },
        fixed_iterations: None,
        energy_threshold: 1e-6,
        boundary_condition: boundary.clone(),
        spline_end_condition: spline_ec,
    };

    let mut cfg = EmdConfig::default();
    cfg.max_imfs = max_imfs;
    cfg.validate_reconstruction = true;
    cfg.sifting_config = sifting_config;
    cfg.boundary_condition = boundary;

    let result = emd(&signal, &cfg)?;
    let collection = &result.imfs;
    eprintln!("Extracted {} IMFs + residual", collection.imfs.len());

    // ── write output CSV ──────────────────────────────────────────────────────
    let mut out = fs::File::create(&output_path)?;

    // Header: residue first, then imf1..imfN
    let mut header: Vec<String> = vec!["residue".to_string()];
    for i in 1..=collection.imfs.len() {
        header.push(format!("imf{}", i));
    }
    writeln!(out, "{}", header.join(","))?;

    let n = signal.len();
    for i in 0..n {
        let residue_val = collection.residue.get(i).copied().unwrap_or(0.0);
        let mut row: Vec<String> = vec![format!("{:.9}", residue_val)];
        for imf in &collection.imfs {
            row.push(format!("{:.9}", imf.get(i).copied().unwrap_or(0.0)));
        }
        writeln!(out, "{}", row.join(","))?;
    }

    eprintln!("Wrote IMFs to {}", output_path);
    Ok(())
}
