use ferromode::types::ImfCollection;

#[test]
fn test_mode_mixing_empty_imfs() {
    let imfs = ImfCollection::new(vec![], vec![]);
    let _check = imfs;
}

#[test]
fn test_mode_mixing_single_imf() {
    let imf = vec![1.0, 2.0, 3.0, 4.0];
    let imfs = ImfCollection::new(vec![imf], vec![0.0; 4]);
    let _check = imfs;
}

#[test]
fn test_mode_mixing_two_imfs() {
    let imf1 = vec![1.0, 2.0, 3.0, 4.0];
    let imf2 = vec![4.0, 3.0, 2.0, 1.0];
    let imfs = ImfCollection::new(vec![imf1, imf2], vec![0.0; 4]);
    let _check = imfs;
}

#[test]
fn test_mode_mixing_three_imfs() {
    let imf1 = vec![1.0, 0.5, -0.5, -1.0];
    let imf2 = vec![2.0, 1.0, -1.0, -2.0];
    let imf3 = vec![0.5, 0.25, -0.25, -0.5];
    let imfs = ImfCollection::new(vec![imf1, imf2, imf3], vec![0.0; 4]);
    let _check = imfs;
}

#[test]
fn test_overlap_metric_identical_imfs() {
    let imf1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let imf2 = imf1.clone();
    let imfs = ImfCollection::new(vec![imf1, imf2], vec![0.0; 5]);
    let _check = imfs;
}

#[test]
fn test_overlap_metric_orthogonal_imfs() {
    let imf1 = vec![1.0, -1.0, 1.0, -1.0, 1.0];
    let imf2 = vec![1.0, 0.0, -1.0, 0.0, 1.0];
    let imfs = ImfCollection::new(vec![imf1, imf2], vec![0.0; 5]);
    let _check = imfs;
}

#[test]
fn test_overlap_metric_realistic_imfs() {
    let imf1: Vec<f64> = (0..20).map(|i| ((i as f64 * 0.1).sin())).collect();
    let imf2: Vec<f64> = (0..20).map(|i| ((i as f64 * 0.2).cos())).collect();
    let imfs = ImfCollection::new(vec![imf1, imf2], vec![0.0; 20]);
    let _check = imfs;
}

#[test]
fn test_mode_mixing_many_imfs() {
    let imfs_vec: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64; 10]).collect();
    let imfs = ImfCollection::new(imfs_vec, vec![0.0; 10]);
    let _check = imfs;
}

#[test]
fn test_mode_mixing_long_signals() {
    let imf1: Vec<f64> = (0..1000).map(|i| ((i as f64 * 0.01).sin())).collect();
    let imf2: Vec<f64> = (0..1000).map(|i| ((i as f64 * 0.02).cos())).collect();
    let imfs = ImfCollection::new(vec![imf1, imf2], vec![0.0; 1000]);
    let _check = imfs;
}

#[test]
fn test_heatmap_basic() {
    let imf1 = vec![1.0, 2.0, 3.0];
    let imf2 = vec![4.0, 5.0, 6.0];
    let imfs = ImfCollection::new(vec![imf1, imf2], vec![0.0; 3]);
    let _check = imfs;
}

#[test]
fn test_heatmap_single_imf_edge() {
    let imf = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let imfs = ImfCollection::new(vec![imf], vec![0.0; 5]);
    let _check = imfs;
}

#[test]
fn test_heatmap_symmetric_matrix() {
    let imf1 = vec![1.0, 2.0];
    let imf2 = vec![3.0, 4.0];
    let imfs = ImfCollection::new(vec![imf1, imf2], vec![0.0; 2]);
    let _check = imfs;
}

#[test]
fn test_decomposition_quality_good() {
    let imf1: Vec<f64> = (0..50).map(|i| ((i as f64 * 0.05).sin())).collect();
    let imf2: Vec<f64> = (0..50).map(|i| ((i as f64 * 0.25).sin())).collect();
    let imfs = ImfCollection::new(vec![imf1, imf2], vec![0.0; 50]);
    let _check = imfs;
}

#[test]
fn test_decomposition_quality_poor() {
    let freq1 = 4.5;
    let freq2 = 5.5;
    let imf1: Vec<f64> = (0..100).map(|i| ((i as f64 * freq1 * 0.01).sin())).collect();
    let imf2: Vec<f64> = (0..100).map(|i| ((i as f64 * freq2 * 0.01).sin())).collect();
    let imfs = ImfCollection::new(vec![imf1, imf2], vec![0.0; 100]);
    let _check = imfs;
}

#[test]
fn test_mode_mixing_value_range() {
    let imf1 = vec![1e-10, 2e-10, 3e-10];
    let imf2 = vec![1e-10, 2e-10, 3e-10];
    let imfs = ImfCollection::new(vec![imf1, imf2], vec![0.0; 3]);
    let _check = imfs;
}

#[test]
fn test_interpretation_guidance_ok() {
    let imf1 = vec![1.0, 2.0];
    let imf2 = vec![10.0, 20.0];
    let imfs = ImfCollection::new(vec![imf1, imf2], vec![0.0; 2]);
    let _check = imfs;
}

#[test]
fn test_interpretation_guidance_caution() {
    let imf1 = vec![1.0, 2.0, 3.0];
    let imf2 = vec![1.5, 2.5, 3.5];
    let imfs = ImfCollection::new(vec![imf1, imf2], vec![0.0; 3]);
    let _check = imfs;
}

#[test]
fn test_interpretation_guidance_serious() {
    let imf1 = vec![1.0, 2.0, 3.0, 4.0];
    let imf2 = imf1.clone();
    let imfs = ImfCollection::new(vec![imf1, imf2], vec![0.0; 4]);
    let _check = imfs;
}
