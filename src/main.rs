mod args;
use crate::args::CommandParse;
use crate::args::Commands;
use clap::Parser;
mod regressor;
use figlet_rs::FIGfont;
use regressor::expression_regressor;
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::metrics::accuracy;
mod knn_kfold;
mod random_kfold;
use crate::knn_kfold::scaledknn;
use crate::random_kfold::scaledrandom;
mod kcross;
mod logcross;
mod rancross;
use crate::kcross::knncross;
use crate::logcross::logisiticcross;
use crate::rancross::randomcross;
mod bceloss;
mod burnneural;
mod neuraltensor;
use burnneural::burnarray;
mod xgboost;
use crate::xgboost::expression_xgboost;

/*
Gaurav Sablok
gsablok@proton.me
*/

fn main() {
    let fontgenerate = FIGfont::standard().unwrap();
    let repgenerate = fontgenerate.convert("httclass");
    println!("{}", repgenerate.unwrap());

    let args = CommandParse::parse();
    match &args.command {
        Commands::RegressorHTT { pathname } => {
            let threads = 4.to_string();
            let threadlaunch = threads.parse::<usize>().unwrap();
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(threadlaunch)
                .build()
                .expect("threads not present");
            value.install(|| {
                let valueunpack = expression_regressor(pathname).unwrap();

                let logistic =
                    LogisticRegression::fit(&valueunpack.0, &valueunpack.2, Default::default())
                        .unwrap();
                let logisticpredict = logistic.predict(&valueunpack.1).unwrap();
                let accuracy = accuracy(&valueunpack.3, &logisticpredict);
                println!("The accuracy of the model is {}", accuracy);
            });
        }
        Commands::KNNHTT { pathname } => {
            let threads = 4.to_string();
            let threadlaunch = threads.parse::<usize>().unwrap();
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(threadlaunch)
                .build()
                .expect("Threads failed");
            value.install(|| {
                let scaledknnvalue = scaledknn(pathname).unwrap();
                println!(
                    "The scaled knn has been completed and the accuracy has been:{}",
                    scaledknnvalue
                );
            });
        }
        Commands::RandomHTT { pathname } => {
            let threads = 4.to_string();
            let threadlaunch = threads.parse::<usize>().unwrap();
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(threadlaunch)
                .build()
                .expect("threads failed to launch");
            value.install(||{
               let scaledrandom = scaledrandom(pathname).unwrap();
               println!("The scaled random classifier on the HTT dataset with the accuracy of the model:{}", scaledrandom);
            });
        }
        Commands::CrossValidateLogistic { pathname } => {
            let threads = 4.to_string();
            let threadslaunch = threads.parse::<usize>().unwrap();
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(threadslaunch)
                .build()
                .expect("threads failed to launch");
            value.install(|| {
                let command = logisiticcross(pathname).unwrap();
                println!("The logistic cross has been completed:{}", command);
            });
        }
        Commands::CrossVaidateKNN { pathname } => {
            let threads = 4.to_string();
            let threadslaunch = threads.parse::<usize>().unwrap();
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(threadslaunch)
                .build()
                .expect("Threads failed to launch");
            value.install(|| {
                let command = knncross(pathname).unwrap();
                println!("The command has been completed:{}", command);
            })
        }
        Commands::CrossValidateRandom { pathname } => {
            let threads = 4.to_string();
            let threadslaunch = threads.parse::<usize>().unwrap();
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(threadslaunch)
                .build()
                .expect("Threads failed to launch");
            value.install(|| {
                let command = randomcross(pathname).unwrap();
                println!("The command has been finished:{}", command);
            })
        }
        Commands::BurnClass { pathname } => {
            let threads = 4.to_string();
            let threadslaunch = threads.parse::<usize>().unwrap();
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(threadslaunch)
                .build()
                .expect("Threads failed to launch");
            value.install(|| {
                let command = burnarray(pathname).unwrap();

                println!(
                    "The command has finished and the neural logistic has been trained:{}",
                    command
                );
            });
        }
        Commands::XGBoost { pathname } => {
            let threads = 4.to_string();
            let threadslaunch = threads.parse::<usize>().unwrap();
            let value = rayon::ThreadPoolBuilder::new()
                .num_threads(threadslaunch)
                .build()
                .expect("Threads failed to launch");
            value.install(|| {
                let command = expression_xgboost(pathname).unwrap();

                println!(
                    "The command has finished and the neural logistic has been trained:{:?}",
                    command
                );
            });
        }
    }
}
