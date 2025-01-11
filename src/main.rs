use csv::ReaderBuilder;
use rand::seq::SliceRandom;
use std::error::Error;
use plotters::prelude::*;

fn read_dataset(file_path: &str) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().from_path(file_path)?;
    let mut features = Vec::new();
    let mut labels = Vec::new();

    for record in reader.records() {
        let record = record?;
        let row: Vec<f64> = record.iter().take(6).map(|val| val.parse::<f64>().unwrap()).collect();
        features.push(row);
        labels.push(record[6].parse::<f64>().unwrap());
    }

    Ok((features, labels))
}

fn normalize(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut transposed = vec![vec![]; data[0].len()];
    for row in data {
        for (i, &value) in row.iter().enumerate() {
            transposed[i].push(value);
        }
    }

    let normalized: Vec<Vec<f64>> = transposed
        .iter()
        .map(|column| {
            let min = column.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = column.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            column.iter().map(|&val| (val - min) / (max - min)).collect()
        })
        .collect();

    (0..data.len())
        .map(|i| normalized.iter().map(|col| col[i]).collect())
        .collect()
}

fn split_data(
    x: &[Vec<f64>],
    y: &[f64],
    train_ratio: f64,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    let mut indices: Vec<usize> = (0..x.len()).collect();
    indices.shuffle(&mut rand::thread_rng());

    let train_size = (train_ratio * x.len() as f64).ceil() as usize;

    let train_indices = &indices[..train_size];
    let test_indices = &indices[train_size..];

    let x_train = train_indices.iter().map(|&i| x[i].clone()).collect();
    let y_train = train_indices.iter().map(|&i| y[i]).collect();
    let x_test = test_indices.iter().map(|&i| x[i].clone()).collect();
    let y_test = test_indices.iter().map(|&i| y[i]).collect();

    (x_train, y_train, x_test, y_test)
}

fn gradient_descent(
    x: &[Vec<f64>],
    y: &[f64],
    learning_rate: f64,
    epochs: usize,
) -> (Vec<f64>, Vec<f64>) {
    let n_features = x[0].len();
    let mut coefficients = vec![0.0; n_features];
    let mut loss_history = Vec::new();

    for _ in 0..epochs {
        let mut gradients = vec![0.0; n_features];
        let mut total_loss = 0.0;

        for (i, row) in x.iter().enumerate() {
            let predicted: f64 = row.iter().zip(&coefficients).map(|(xi, coef)| xi * coef).sum();
            let error = predicted - y[i];
            total_loss += error.powi(2);

            for j in 0..n_features {
                gradients[j] += error * row[j];
            }
        }

        for j in 0..n_features {
            coefficients[j] -= learning_rate * gradients[j] / x.len() as f64;
        }

        loss_history.push(total_loss / x.len() as f64);
    }

    (coefficients, loss_history)
}

fn evaluate_model(
    x: &[Vec<f64>],
    y: &[f64],
    coefficients: &[f64],
) -> (Vec<f64>, f64, f64) {
    let predictions: Vec<f64> = x
        .iter()
        .map(|row| {
            row.iter()
                .zip(coefficients)
                .map(|(xi, coef)| xi * coef)
                .sum()
        })
        .collect();

    let mse = y
        .iter()
        .zip(&predictions)
        .map(|(yi, pred)| (yi - pred).powi(2))
        .sum::<f64>()
        / y.len() as f64;

    let mean_y = y.iter().sum::<f64>() / y.len() as f64;
    let ss_total = y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>();
    let ss_residual = y
        .iter()
        .zip(&predictions)
        .map(|(yi, pred)| (yi - pred).powi(2))
        .sum::<f64>();

    let r2 = 1.0 - (ss_residual / ss_total);

    (predictions, mse, r2)
}

fn plot_loss(loss_history: &[f64]) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("loss_plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Loss History", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..loss_history.len(), 0.0..loss_history.iter().cloned().fold(0.0, f64::max))?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            loss_history.iter().enumerate().map(|(i, &loss)| (i, loss)),
            &RED,
        ))?
        .label("Loss")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 10, y)], &RED));

    chart.configure_series_labels().background_style(&WHITE).draw()?;

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let (x, y) = read_dataset("D:/405 FOUND/Linear_Regression-RUST/src/human_zombie_dataset_v5.csv")?;
    let x_normalized = normalize(&x);
    let (x_train, y_train, x_test, y_test) = split_data(&x_normalized, &y, 0.8);

    let learning_rate = 0.001;
    let epochs = 15000;

    let (coefficients, loss_history) = gradient_descent(&x_train, &y_train, learning_rate, epochs);
    plot_loss(&loss_history)?;

    let (_, mse, r2) = evaluate_model(&x_test, &y_test, &coefficients);

    println!("Coefficients: {:?}", coefficients);
    println!("Mean Squared Error: {:.4}", mse);
    println!("R² Score: {:.4}", r2);

    Ok(())
}
