use rayon::prelude::*;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use polynomen::Poly;

#[pymodule]
fn _rl_rust(m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_poly, m)?)?;
    Ok(())
}

fn solve_for_coeffs(coeffs: Vec<f64>) -> Vec<(f64, f64)> {
    let p = Poly::new_from_coeffs(&coeffs);
    let complex_roots = p.complex_roots();
    complex_roots
}

#[pyfunction]
pub fn solve_poly(coeffs_list: Vec<Vec<f64>>) -> Vec<Vec<(f64, f64)>> {
    let complex_roots: Vec<Vec<(f64, f64)>> = coeffs_list.par_iter().map(|coeffs| solve_for_coeffs((&coeffs).to_vec())).collect();
    complex_roots
}