use smartcore::api::{Transformer, UnsupervisedEstimator};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::accuracy;
use smartcore::model_selection::train_test_split;
use smartcore::neighbors::knn_classifier::KNNClassifier;
use smartcore::preprocessing::numerical::StandardScaler;
use smartcore::preprocessing::numerical::StandardScalerParameters;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

/*
Gaurav Sablok
gsablok@proton.me
*/

pub fn scaledknn(pathfile: &str) -> Result<String, Box<dyn Error>> {
    let mut control1: Vec<f32> = Vec::new();
    let mut control2: Vec<f32> = Vec::new();
    let mut control3: Vec<f32> = Vec::new();

    let mut replicate1_1: Vec<f32> = Vec::new();
    let mut replicate1_2: Vec<f32> = Vec::new();
    let mut replicate1_3: Vec<f32> = Vec::new();

    let mut replicate2_1: Vec<f32> = Vec::new();
    let mut replicate2_2: Vec<f32> = Vec::new();
    let mut replicate2_3: Vec<f32> = Vec::new();

    let mut replicate3_1: Vec<f32> = Vec::new();
    let mut replicate3_2: Vec<f32> = Vec::new();
    let mut replicate3_3: Vec<f32> = Vec::new();

    let fileopen = File::open(pathfile).expect("file not present");
    let fileread = BufReader::new(fileopen);

    for i in fileread.lines() {
        let line = i.expect("file not present");
        let linesplit = line
            .split(",")
            .map(|x| x.parse::<f32>().unwrap())
            .collect::<Vec<_>>();
        control1.push(linesplit[0]);
        control2.push(linesplit[1]);
        control3.push(linesplit[3]);
        replicate1_1.push(linesplit[4]);
        replicate1_2.push(linesplit[5]);
        replicate1_3.push(linesplit[6]);
        replicate2_1.push(linesplit[7]);
        replicate2_2.push(linesplit[8]);
        replicate2_3.push(linesplit[9]);
        replicate3_1.push(linesplit[10]);
        replicate3_2.push(linesplit[11]);
        replicate3_3.push(linesplit[12]);
    }

    /*
    Math summing the value across the expression and creating a f64 tuple for the expression
    then using that to classify the samples using the control measure as that is a standard
    expression and since this is a classification, so that will introduce no bias so
    the math will be no biased such as wx+b and no sigmoid needed.
    */

    let control_add = control1
        .iter()
        .zip(control2)
        .zip(control3)
        .map(|x| x.0.0 + x.0.1 + x.1)
        .collect::<Vec<f32>>()
        .iter()
        .map(|x| x.to_string().parse::<i32>().unwrap())
        .collect::<Vec<_>>();

    let replicate_1_add = replicate1_1
        .iter()
        .zip(replicate1_2)
        .zip(replicate1_3)
        .map(|x| x.0.0 + x.0.1 + x.1)
        .collect::<Vec<f32>>()
        .iter()
        .map(|x| x.to_string().parse::<i32>().unwrap())
        .collect::<Vec<_>>();

    let replicate_2_add = replicate2_1
        .iter()
        .zip(replicate2_2)
        .zip(replicate2_3)
        .map(|x| x.0.0 + x.0.1 + x.1)
        .collect::<Vec<_>>()
        .iter()
        .map(|x| x.to_string().parse::<i32>().unwrap())
        .collect::<Vec<_>>();

    let replicate_3_add = replicate3_1
        .iter()
        .zip(replicate3_2)
        .zip(replicate3_3)
        .map(|x| x.0.0 + x.0.1 + x.1)
        .collect::<Vec<_>>()
        .iter()
        .map(|x| x.to_string().parse::<i32>().unwrap())
        .collect::<Vec<_>>();

    let control_mean = control_add.iter().max().unwrap()
        + control_add.iter().min().unwrap() / control_add.len().to_string().parse::<i32>().unwrap();

    let _replicate1_mean = replicate_1_add.iter().max().unwrap()
        + replicate_1_add.iter().min().unwrap()
            / replicate_1_add.len().to_string().parse::<i32>().unwrap();

    let _replicate2_mean = replicate_2_add.iter().max().unwrap()
        + replicate_2_add.iter().min().unwrap()
            / replicate_2_add.len().to_string().parse::<i32>().unwrap();

    let _replicate3_mean = replicate_3_add.iter().max().unwrap()
        + replicate_3_add.iter().min().unwrap()
            / replicate_3_add
                .iter()
                .min()
                .unwrap()
                .to_string()
                .parse::<i32>()
                .unwrap();

    let expression_tuple = control_add
        .iter()
        .zip(replicate_1_add)
        .zip(replicate_2_add)
        .zip(replicate_3_add)
        .map(|x| vec![(*x.0.0.0, x.0.0.1, x.0.1, x.1)])
        .collect::<Vec<_>>();

    let labels: Vec<i32> = expression_tuple
        .iter()
        .map(|x| if x[0].0 <= control_mean { 0 } else { 1 })
        .collect::<Vec<_>>();

    let mut densematrix: Vec<Vec<f64>> = Vec::new();
    for i in expression_tuple.iter() {
        for j in i.iter() {
            densematrix.push(vec![
                j.0.to_string().parse::<f64>().unwrap(),
                j.1.to_string().parse::<f64>().unwrap(),
                j.2.to_string().parse::<f64>().unwrap(),
                j.3.to_string().parse::<f64>().unwrap(),
            ]);
        }
    }

    let densematrix_sep = DenseMatrix::from_2d_vec(&densematrix);

    let trainsplit = train_test_split(&densematrix_sep, &labels, 0.2, true, Some(2811));

    let trainsplit_1 = trainsplit.0;
    let trainsplit_2 = trainsplit.1;
    let trainsplit_3 = trainsplit.2;
    let trainsplit_4 = trainsplit.3;

    let scaler =
        StandardScaler::fit(&densematrix_sep, StandardScalerParameters::default()).unwrap();

    let scaled = scaler.transform(&trainsplit_1).unwrap();

    let knnclassifier = KNNClassifier::fit(&scaled, &trainsplit_3, Default::default()).unwrap();

    let predicted = knnclassifier.predict(&trainsplit_2).unwrap();

    let accuracy_knn = accuracy(&trainsplit_4, &predicted);

    println!("The prediction accuracy of the model is {}", accuracy_knn);

    Ok("Knn scaled classifier has been completed".to_string())
}
