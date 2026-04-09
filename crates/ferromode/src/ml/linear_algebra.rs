#![warn(missing_docs)]

//! Linear algebra utilities for implicit differentiation.
//!
//! Provides matrix operations, decompositions, and linear system solvers
//! for computing implicit gradients via Jacobian inversion.

use crate::error::EmdError;
use std::fmt;

// ---------------------------------------------------------------------------
// Matrix Type
// ---------------------------------------------------------------------------

/// A dense 2D matrix stored in row-major order.
#[derive(Debug, Clone)]
pub struct Matrix {
    /// Matrix data in row-major order
    data: Vec<f64>,
    /// Number of rows
    rows: usize,
    /// Number of columns
    cols: usize,
}

impl Matrix {
    /// Create a new matrix with given dimensions, initialized to zero.
    ///
    /// # Arguments
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    ///
    /// # Returns
    /// A new zero-initialized matrix of shape (rows, cols)
    ///
    /// # Example
    /// ```
    /// use ferromode::ml::linear_algebra::Matrix;
    /// let m = Matrix::zeros(3, 3);
    /// assert_eq!(m.shape(), (3, 3));
    /// ```
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { data: vec![0.0; rows * cols], rows, cols }
    }

    /// Create identity matrix of given size.
    ///
    /// # Arguments
    /// * `n` - Matrix dimension (creates n×n identity matrix)
    ///
    /// # Example
    /// ```
    /// use ferromode::ml::linear_algebra::Matrix;
    /// let i = Matrix::eye(3);
    /// assert_eq!(i.shape(), (3, 3));
    /// ```
    pub fn eye(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    /// Get matrix dimensions as (rows, cols).
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get value at (row, col).
    ///
    /// # Panics
    /// Panics if indices are out of bounds.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        self.data[row * self.cols + col]
    }

    /// Set value at (row, col).
    ///
    /// # Panics
    /// Panics if indices are out of bounds.
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        self.data[row * self.cols + col] = value;
    }

    /// Get mutable reference to value at (row, col).
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut f64 {
        assert!(row < self.rows && col < self.cols, "index out of bounds");
        &mut self.data[row * self.cols + col]
    }

    /// Get raw data slice.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Get mutable raw data slice.
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.cols
    }

    /// Transpose this matrix (creates new matrix).
    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    /// Matrix-vector multiplication: self @ vec
    ///
    /// # Errors
    /// Returns an error if dimensions don't match.
    pub fn matvec(&self, vec: &[f64]) -> Result<Vec<f64>, EmdError> {
        if vec.len() != self.cols {
            return Err(EmdError::InvalidConfig(format!(
                "dimension mismatch in matvec: {} != {}",
                vec.len(),
                self.cols
            )));
        }

        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self.get(i, j) * vec[j];
            }
        }
        Ok(result)
    }

    /// Matrix-matrix multiplication: self @ other
    ///
    /// # Errors
    /// Returns an error if dimensions don't match.
    pub fn matmul(&self, other: &Matrix) -> Result<Matrix, EmdError> {
        if self.cols != other.rows {
            return Err(EmdError::InvalidConfig(format!(
                "dimension mismatch in matmul: ({}, {}) @ ({}, {})",
                self.rows, self.cols, other.rows, other.cols
            )));
        }

        let mut result = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let aik = self.get(i, k);
                for j in 0..other.cols {
                    let val = result.get(i, j) + aik * other.get(k, j);
                    result.set(i, j, val);
                }
            }
        }
        Ok(result)
    }

    /// Frobenius norm: sqrt(sum(A_ij²))
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Maximum absolute value in matrix.
    pub fn max_abs(&self) -> f64 {
        self.data.iter().map(|x| x.abs()).fold(0.0, f64::max)
    }

    /// Clone the matrix.
    pub fn clone_matrix(&self) -> Matrix {
        Matrix { data: self.data.clone(), rows: self.rows, cols: self.cols }
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Matrix({} x {})", self.rows, self.cols)?;
        for i in 0..self.rows {
            write!(f, "[")?;
            for j in 0..self.cols {
                write!(f, "{:9.4} ", self.get(i, j))?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LU Decomposition
// ---------------------------------------------------------------------------

/// Result of LU decomposition: (L, U) where A = L @ U
pub struct LuDecomposition {
    /// Lower triangular matrix (with implicit 1s on diagonal)
    l: Matrix,
    /// Upper triangular matrix
    u: Matrix,
    /// Pivot indices for row permutations
    pivots: Vec<usize>,
}

impl LuDecomposition {
    /// Get the L factor (lower triangular with 1s on diagonal).
    pub fn l(&self) -> &Matrix {
        &self.l
    }

    /// Get the U factor (upper triangular).
    pub fn u(&self) -> &Matrix {
        &self.u
    }

    /// Get pivot indices.
    pub fn pivots(&self) -> &[usize] {
        &self.pivots
    }
}

/// Compute LU decomposition of A with partial pivoting.
///
/// Returns `LuDecomposition` containing L and U factors such that P @ A = L @ U,
/// where P is the permutation matrix encoded in `pivots`.
///
/// # Errors
/// Returns an error if the matrix is singular or nearly singular.
///
/// # Algorithm
/// Standard Gaussian elimination with partial pivoting for numerical stability.
pub fn lu_decomposition(a: &Matrix) -> Result<LuDecomposition, EmdError> {
    let (m, n) = a.shape();
    if m == 0 || n == 0 {
        return Err(EmdError::InvalidConfig("empty matrix".to_string()));
    }

    let mut l = Matrix::zeros(m, m);
    let mut u = a.clone_matrix();
    let mut pivots = (0..m).collect::<Vec<_>>();

    // Fill lower triangular part with 1s on diagonal
    for i in 0..m {
        l.set(i, i, 1.0);
    }

    // LU decomposition with partial pivoting
    for k in 0..std::cmp::min(m, n) {
        // Find pivot
        let mut pivot_row = k;
        let mut pivot_val = u.get(k, k).abs();
        for i in (k + 1)..m {
            let val = u.get(i, k).abs();
            if val > pivot_val {
                pivot_val = val;
                pivot_row = i;
            }
        }

        if pivot_val < 1e-15 {
            return Err(EmdError::InvalidConfig(
                "matrix is singular or nearly singular (pivot < 1e-15)".to_string(),
            ));
        }

        // Swap rows in L and U
        if pivot_row != k {
            for j in 0..n {
                let tmp = u.get(k, j);
                u.set(k, j, u.get(pivot_row, j));
                u.set(pivot_row, j, tmp);
            }
            for j in 0..k {
                let tmp = l.get(k, j);
                l.set(k, j, l.get(pivot_row, j));
                l.set(pivot_row, j, tmp);
            }
            pivots.swap(k, pivot_row);
        }

        // Elimination
        let ukk = u.get(k, k);
        for i in (k + 1)..m {
            let lik = u.get(i, k) / ukk;
            l.set(i, k, lik);
            for j in k..n {
                let val = u.get(i, j) - lik * u.get(k, j);
                u.set(i, j, val);
            }
        }
    }

    Ok(LuDecomposition { l, u, pivots })
}

// ---------------------------------------------------------------------------
// Triangular System Solvers
// ---------------------------------------------------------------------------

/// Solve lower triangular system L @ x = b using forward substitution.
///
/// Assumes L has 1s on the diagonal (implicit).
///
/// # Errors
/// Returns an error if dimensions don't match.
pub fn solve_lower_triangular(l: &Matrix, b: &[f64]) -> Result<Vec<f64>, EmdError> {
    let n = l.nrows();
    if l.ncols() != n || b.len() != n {
        return Err(EmdError::InvalidConfig(
            "dimension mismatch in solve_lower_triangular".to_string(),
        ));
    }

    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l.get(i, j) * x[j];
        }
        x[i] = sum; // Diagonal is 1
    }
    Ok(x)
}

/// Solve upper triangular system U @ x = b using back substitution.
///
/// # Errors
/// Returns an error if the matrix is singular or dimensions don't match.
pub fn solve_upper_triangular(u: &Matrix, b: &[f64]) -> Result<Vec<f64>, EmdError> {
    let n = u.nrows();
    if u.ncols() != n || b.len() != n {
        return Err(EmdError::InvalidConfig(
            "dimension mismatch in solve_upper_triangular".to_string(),
        ));
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= u.get(i, j) * x[j];
        }
        let uii = u.get(i, i);
        if uii.abs() < 1e-15 {
            return Err(EmdError::InvalidConfig("upper triangular matrix is singular".to_string()));
        }
        x[i] = sum / uii;
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// Linear System Solver
// ---------------------------------------------------------------------------

/// Solve linear system A @ x = b using LU decomposition.
///
/// # Arguments
/// * `a` - Coefficient matrix (should be square)
/// * `b` - Right-hand side vector
///
/// # Errors
/// Returns an error if the matrix is singular or dimensions don't match.
///
/// # Algorithm
/// 1. Compute LU decomposition of A
/// 2. Solve L @ y = P @ b (forward substitution)
/// 3. Solve U @ x = y (back substitution)
///
/// # Example
/// ```
/// use ferromode::ml::linear_algebra::{Matrix, solve_linear_system};
///
/// // Create a simple 2x2 system: [2, 1] @ [x, y]^T = [5, 4]^T
/// let mut a = Matrix::zeros(2, 2);
/// a.set(0, 0, 2.0);
/// a.set(0, 1, 1.0);
/// a.set(1, 0, 1.0);
/// a.set(1, 1, 3.0);
///
/// let b = vec![5.0, 4.0];
/// let x = solve_linear_system(&a, &b).unwrap();
/// // x ≈ [1.8, 1.4]
/// ```
pub fn solve_linear_system(a: &Matrix, b: &[f64]) -> Result<Vec<f64>, EmdError> {
    let (m, n) = a.shape();
    if m != n {
        return Err(EmdError::InvalidConfig(format!(
            "matrix must be square for linear system solve: {}x{}",
            m, n
        )));
    }
    if b.len() != n {
        return Err(EmdError::InvalidConfig(format!(
            "RHS dimension mismatch: {} != {}",
            b.len(),
            n
        )));
    }

    // Compute LU decomposition
    let lu = lu_decomposition(a)?;

    // Apply pivots to b: b_perm = P @ b
    let mut b_perm = b.to_vec();
    for i in 0..m {
        b_perm[i] = b[lu.pivots[i]];
    }

    // Solve L @ y = b_perm
    let y = solve_lower_triangular(lu.l(), &b_perm)?;

    // Solve U @ x = y
    let x = solve_upper_triangular(lu.u(), &y)?;

    Ok(x)
}

// ---------------------------------------------------------------------------
// Condition Number Estimation
// ---------------------------------------------------------------------------

/// Estimate condition number of a matrix using Frobenius norm.
///
/// For matrices with condition number > 1e10, consider regularization.
///
/// # Errors
/// Returns an error if matrix inversion fails.
pub fn condition_number(a: &Matrix) -> Result<f64, EmdError> {
    let (m, n) = a.shape();
    if m != n {
        return Err(EmdError::InvalidConfig(
            "condition number only defined for square matrices".to_string(),
        ));
    }

    let a_inv = matrix_inverse(a)?;
    let norm_a = a.frobenius_norm();
    let norm_a_inv = a_inv.frobenius_norm();

    Ok(norm_a * norm_a_inv)
}

/// Invert a square matrix using LU decomposition.
///
/// # Errors
/// Returns an error if the matrix is singular.
///
/// # Algorithm
/// For each column e_i of the identity matrix, solve A @ x_i = e_i,
/// and stack results to form A^{-1}.
pub fn matrix_inverse(a: &Matrix) -> Result<Matrix, EmdError> {
    let (m, n) = a.shape();
    if m != n {
        return Err(EmdError::InvalidConfig(format!(
            "matrix must be square for inversion: {}x{}",
            m, n
        )));
    }

    // Compute LU decomposition once
    let lu = lu_decomposition(a)?;

    let mut a_inv = Matrix::zeros(n, n);

    // Solve for each column of identity matrix
    for col in 0..n {
        let mut e = vec![0.0; n];
        e[col] = 1.0;

        // Apply pivots to e: e_perm = P @ e
        let mut e_perm = vec![0.0; n];
        for i in 0..n {
            e_perm[i] = e[lu.pivots[i]];
        }

        // Solve L @ y = e_perm
        let y = solve_lower_triangular(lu.l(), &e_perm)?;

        // Solve U @ x = y
        let x = solve_upper_triangular(lu.u(), &y)?;

        // Store result in column
        for row in 0..n {
            a_inv.set(row, col, x[row]);
        }
    }

    Ok(a_inv)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation_and_access() {
        let mut m = Matrix::zeros(2, 3);
        m.set(0, 0, 1.5);
        m.set(1, 2, 2.5);
        assert_eq!(m.get(0, 0), 1.5);
        assert_eq!(m.get(1, 2), 2.5);
        assert_eq!(m.shape(), (2, 3));
    }

    #[test]
    fn test_matrix_transpose() {
        let mut m = Matrix::zeros(2, 3);
        m.set(0, 1, 1.5);
        m.set(1, 2, 2.5);
        let mt = m.transpose();
        assert_eq!(mt.shape(), (3, 2));
        assert_eq!(mt.get(1, 0), 1.5);
        assert_eq!(mt.get(2, 1), 2.5);
    }

    #[test]
    fn test_matvec() {
        let mut a = Matrix::zeros(2, 3);
        a.set(0, 0, 1.0);
        a.set(0, 1, 2.0);
        a.set(0, 2, 3.0);
        a.set(1, 0, 4.0);
        a.set(1, 1, 5.0);
        a.set(1, 2, 6.0);

        let v = vec![1.0, 2.0, 3.0];
        let result = a.matvec(&v).unwrap();
        assert!((result[0] - 14.0).abs() < 1e-10); // 1*1 + 2*2 + 3*3
        assert!((result[1] - 32.0).abs() < 1e-10); // 4*1 + 5*2 + 6*3
    }

    #[test]
    fn test_solve_linear_system_2x2() {
        // [2 1] [x]   [5]
        // [1 3] [y] = [4]
        // Solution: [2.2, 0.6]
        // Verification: 2*2.2 + 1*0.6 = 4.4 + 0.6 = 5 ✓
        //               1*2.2 + 3*0.6 = 2.2 + 1.8 = 4 ✓
        let mut a = Matrix::zeros(2, 2);
        a.set(0, 0, 2.0);
        a.set(0, 1, 1.0);
        a.set(1, 0, 1.0);
        a.set(1, 1, 3.0);

        let b = vec![5.0, 4.0];
        let x = solve_linear_system(&a, &b).unwrap();

        // Verify by computing A @ x
        let reconstructed = a.matvec(&x).unwrap();
        for (i, (&expected, &computed)) in b.iter().zip(reconstructed.iter()).enumerate() {
            assert!(
                (computed - expected).abs() < 1e-8,
                "mismatch at index {}: computed {}, expected {}",
                i,
                computed,
                expected
            );
        }
    }

    #[test]
    fn test_solve_linear_system_3x3() {
        // [1 2 3] [x]   [6]
        // [0 1 4] [y] = [5]
        // [5 6 0] [z]   [11]
        let mut a = Matrix::zeros(3, 3);
        a.set(0, 0, 1.0);
        a.set(0, 1, 2.0);
        a.set(0, 2, 3.0);
        a.set(1, 0, 0.0);
        a.set(1, 1, 1.0);
        a.set(1, 2, 4.0);
        a.set(2, 0, 5.0);
        a.set(2, 1, 6.0);
        a.set(2, 2, 0.0);

        let b = vec![6.0, 5.0, 11.0];
        let x = solve_linear_system(&a, &b).unwrap();

        // Verify by computing A @ x = b
        let result = a.matvec(&x).unwrap();
        for i in 0..3 {
            assert!((result[i] - b[i]).abs() < 1e-10, "verification failed at index {}", i);
        }
    }

    #[test]
    fn test_lu_decomposition() {
        let mut a = Matrix::zeros(3, 3);
        a.set(0, 0, 4.0);
        a.set(0, 1, 3.0);
        a.set(1, 0, 6.0);
        a.set(1, 1, 3.0);
        a.set(2, 0, -2.0);
        a.set(2, 1, 5.0);
        a.set(2, 2, 2.0);

        let lu = lu_decomposition(&a).unwrap();
        assert_eq!(lu.l().shape(), (3, 3));
        assert_eq!(lu.u().shape(), (3, 3));
    }

    #[test]
    fn test_singular_matrix_detection() {
        // Singular matrix: rows are linearly dependent
        let mut a = Matrix::zeros(2, 2);
        a.set(0, 0, 1.0);
        a.set(0, 1, 2.0);
        a.set(1, 0, 2.0);
        a.set(1, 1, 4.0);

        let result = solve_linear_system(&a, &vec![1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_condition_number() {
        let mut a = Matrix::zeros(2, 2);
        a.set(0, 0, 2.0);
        a.set(0, 1, 1.0);
        a.set(1, 0, 1.0);
        a.set(1, 1, 3.0);

        let cond = condition_number(&a).unwrap();
        assert!(cond > 0.0);
        assert!(cond < 100.0); // Should be well-conditioned
    }

    #[test]
    fn test_matrix_inverse() {
        let mut a = Matrix::zeros(2, 2);
        a.set(0, 0, 2.0);
        a.set(0, 1, 1.0);
        a.set(1, 0, 1.0);
        a.set(1, 1, 3.0);

        let a_inv = matrix_inverse(&a).unwrap();

        // Compute A @ A^{-1} (should be identity)
        let ident = a.matmul(&a_inv).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((ident.get(i, j) - expected).abs() < 1e-10);
            }
        }
    }
}
