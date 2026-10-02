use clap::{Parser, Subcommand};
#[derive(Debug, Parser)]
#[command(
    name = "httclass",
    version = "1.0",
    about = "Regression Modelling for HTT Gene
       ************************************************
       Gaurav Sablok,
       Email: gsablok@proton.me
      ************************************************"
)]
pub struct CommandParse {
    /// subcommands for the specific actions
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Regression Modelling for HTT
    RegressorHTT {
        /// path to the file
        pathname: String,
    },
    /// KNN scaler modelling
    KNNHTT {
        /// path to the file
        pathname: String,
    },
    /// Random Scaler Modelling
    RandomHTT {
        /// path to the file
        pathname: String,
    },
    /// Logistic Cross Validate the dataset
    CrossValidateLogistic {
        /// path to the file
        pathname: String,
    },
    /// Random Forest cross validate the dataset
    CrossValidateRandom {
        /// path to the file
        pathname: String,
    },
    /// KNN cross validate the dataset
    CrossVaidateKNN {
        /// path to the file
        pathname: String,
    },
    /// Neural Burn classification
    BurnClass {
        /// path to the filename
        pathname: String,
    },
    /// XGBoost classification
    XGBoost {
        /// path to the filename
        pathname: String,
    },
}
