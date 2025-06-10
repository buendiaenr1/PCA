use csv::ReaderBuilder;
use plotters::prelude::*;
use smartcore::decomposition::pca::{PCA, PCAParameters};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;

fn read_csv(file_path: &str) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn std::error::Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(file_path)?;
    let mut data: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<f64> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let row: Vec<f64> = record.iter().take(record.len() - 1)
            .map(|s| s.parse().unwrap())
            .collect();
        let target: f64 = record[record.len() - 1].parse().unwrap();
        data.push(row);
        targets.push(target);
    }

    Ok((data, targets))
}

fn perform_pca(data: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
    // Check for empty data or inconsistent row lengths
    if data.is_empty() {
        return Err("No data provided".into());
    }
    let n_features = data[0].len();
    if !data.iter().all(|row| row.len() == n_features) {
        return Err("Inconsistent number of features in data".into());
    }

    // Convert to DenseMatrix (column-major)
    let n_samples = data.len();
    let flat_data: Vec<f64> = data.iter().flatten().cloned().collect();
    let x = DenseMatrix::new(n_samples, n_features, flat_data,false);

    // Perform PCA
    let pca = PCA::fit(&x, PCAParameters::default().with_n_components(2))?;
    
    // Transform data
    let result = pca.transform(&x)?;
    
    // Convert back to Vec<Vec<f64>>
    let mut projected_data = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut row = Vec::with_capacity(2);
        for j in 0..2 {
            row.push(*result.get((i, j)));
        }
        projected_data.push(row);
    }

    Ok(projected_data)
}

fn plot_pca_results(projected_data: &[Vec<f64>], targets: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("pca_results.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Calculate plot bounds
    let x_min = projected_data.iter().map(|v| v[0]).fold(f64::INFINITY, |a, b| a.min(b)) - 1.0;
    let x_max = projected_data.iter().map(|v| v[0]).fold(f64::NEG_INFINITY, |a, b| a.max(b)) + 1.0;
    let y_min = projected_data.iter().map(|v| v[1]).fold(f64::INFINITY, |a, b| a.min(b)) - 1.0;
    let y_max = projected_data.iter().map(|v| v[1]).fold(f64::NEG_INFINITY, |a, b| a.max(b)) + 1.0;

    let mut chart = ChartBuilder::on(&root)
        .caption("PCA Results", ("sans-serif", 40).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart.configure_mesh().draw()?;

    for (i, point) in projected_data.iter().enumerate() {
        let color = if targets[i] == 1.0 { RED } else { BLUE };
        chart.draw_series(std::iter::once(
            Circle::new(
                (point[0], point[1]),
                5,
                ShapeStyle::from(&color).filled(),
            )
        ))?;
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = "data.csv";
    let (data, targets) = read_csv(file_path)?;
    let projected_data = perform_pca(&data)?;
    plot_pca_results(&projected_data, &targets)?;
    println!("PCA completed successfully. Results saved to pca_results.png");
    Ok(())
}