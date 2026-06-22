//! Exploration harness: characterize the true Jacobian d(IMF)/d(signal) of the
//! EMD forward via finite differences, to validate the "EMD is locally exactly
//! linear at its realized extrema configuration" hypothesis before building the
//! analytic backward pass.
//!
//! Run: cargo run -p ferromode --example diff_groundtruth

use ferromode::algorithms::emd::{emd, EmdConfig};

fn test_signal(n: usize) -> Vec<f64> {
    use std::f64::consts::PI;
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 12.0 * t).sin()
        })
        .collect()
}

/// First IMF as a function of the signal (panics on failure — exploration only).
fn imf0(signal: &[f64], cfg: &EmdConfig) -> Vec<f64> {
    let r = emd(signal, cfg).expect("emd");
    r.imfs.imfs[0].clone()
}

/// Finite-difference Jacobian J[i][j] = d c0_i / d x_j (central difference).
fn fd_jacobian(signal: &[f64], cfg: &EmdConfig, eps: f64) -> Vec<Vec<f64>> {
    let n = signal.len();
    let mut j = vec![vec![0.0; n]; n];
    for col in 0..n {
        let mut sp = signal.to_vec();
        sp[col] += eps;
        let cp = imf0(&sp, cfg);
        let mut sm = signal.to_vec();
        sm[col] -= eps;
        let cm = imf0(&sm, cfg);
        for row in 0..n {
            j[row][col] = (cp[row] - cm[row]) / (2.0 * eps);
        }
    }
    j
}

fn main() {
    let n = 64;
    let cfg = EmdConfig { max_imfs: 1, ..Default::default() };
    let signal = test_signal(n);

    // 1) Linearity check: does c0(x + a·d) - c0(x) scale linearly in a for a
    //    random direction d? If EMD is locally linear, the response is exactly
    //    proportional to a (until extrema flip).
    let c0 = imf0(&signal, &cfg);
    let dir: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 5) as f64 - 2.0).collect();
    println!("== Linearity of c0(x + a*d) - c0(x) ==");
    let mut base_ratio: Option<f64> = None;
    for &a in &[1e-6, 1e-5, 1e-4, 1e-3, 1e-2] {
        let xp: Vec<f64> = signal.iter().zip(&dir).map(|(&s, &d)| s + a * d).collect();
        let cp = imf0(&xp, &cfg);
        let diff_norm: f64 =
            cp.iter().zip(&c0).map(|(&p, &b)| (p - b).powi(2)).sum::<f64>().sqrt();
        let ratio = diff_norm / a;
        let dev = base_ratio.map_or(0.0, |b| ((ratio - b) / b).abs());
        if base_ratio.is_none() {
            base_ratio = Some(ratio);
        }
        println!("  a={a:.0e}  ||Δc0||/a = {ratio:.6e}   rel-dev-from-first={dev:.2e}");
    }

    // 2) Jacobian stability across step sizes: if the map is locally linear, the
    //    FD Jacobian is (nearly) independent of eps.
    println!("\n== FD Jacobian eps-stability (Frobenius norm + max abs) ==");
    let mut prev: Option<Vec<Vec<f64>>> = None;
    for &eps in &[1e-3, 1e-4, 1e-5] {
        let j = fd_jacobian(&signal, &cfg, eps);
        let fro: f64 = j.iter().flatten().map(|v| v * v).sum::<f64>().sqrt();
        let maxabs = j.iter().flatten().fold(0.0f64, |m, &v| m.max(v.abs()));
        let drift = prev.as_ref().map(|p| {
            let mut d = 0.0f64;
            for r in 0..n {
                for c in 0..n {
                    d = d.max((j[r][c] - p[r][c]).abs());
                }
            }
            d
        });
        println!(
            "  eps={eps:.0e}  ||J||_F={fro:.4}  max|J|={maxabs:.4}  max-drift-vs-prev={}",
            drift.map_or_else(|| "-".into(), |d| format!("{d:.2e}"))
        );
        prev = Some(j);
    }

    // 3) Row-sum check: a sifting step is h - mean_env(h); mean_env of a constant
    //    is that constant, so (I - P) annihilates constants => each IMF has ~zero
    //    response to a uniform shift => Jacobian row sums ~ 0.
    let j = fd_jacobian(&signal, &cfg, 1e-4);
    let mut max_rowsum = 0.0f64;
    for row in &j {
        max_rowsum = max_rowsum.max(row.iter().sum::<f64>().abs());
    }
    println!("\n== Jacobian row-sum (expect ~0: IMF ignores constant offset) ==");
    println!("  max |row sum| = {max_rowsum:.3e}");

    // 4) Sum-of-IMFs+residue Jacobian should be identity (reconstruction is exact):
    //    d(Σc + r)/dx = I. Verify d(Σc+r)_i/dx_j ≈ δ_ij via one column.
    let cfg_all = EmdConfig { max_imfs: 0, ..Default::default() };
    let recon = |x: &[f64]| -> Vec<f64> {
        let r = emd(x, &cfg_all).expect("emd");
        let mut s = r.imfs.residue.clone();
        for imf in &r.imfs.imfs {
            for (i, v) in imf.iter().enumerate() {
                s[i] += v;
            }
        }
        s
    };
    let eps = 1e-5;
    let col = n / 2;
    let mut xp = signal.clone();
    xp[col] += eps;
    let mut xm = signal.clone();
    xm[col] -= eps;
    let rp = recon(&xp);
    let rm = recon(&xm);
    let mut max_off = 0.0f64;
    let mut diag = 0.0;
    for i in 0..n {
        let d = (rp[i] - rm[i]) / (2.0 * eps);
        if i == col {
            diag = d;
        } else {
            max_off = max_off.max(d.abs());
        }
    }
    println!("\n== Reconstruction Jacobian column (expect e_col) ==");
    println!("  diag={diag:.6}  max off-diag={max_off:.3e}");
}
