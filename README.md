## httclass

- a specific expression based regressor on RNA-seq with modified binding site for HTT disease.
- from classification to neural classification to cross-validation for the HTT specific data.
- Publication with the experimenttal data is under preparation.
- k-fold cross validation and all classification with math implemented as mean of variance.



```
cargo build
```

```
  _       _     _            _                     
 | |__   | |_  | |_    ___  | |   __ _   ___   ___ 
 | '_ \  | __| | __|  / __| | |  / _` | / __| / __|
 | | | | | |_  | |_  | (__  | | | (_| | \__ \ \__ \
 |_| |_|  \__|  \__|  \___| |_|  \__,_| |___/ |___/
                                                   

Regression Modelling for HTT Gene
       ************************************************
       Gaurav Sablok,
       Email: gsablok@proton.me
      ************************************************

Usage: httclass <COMMAND>

Commands:
  regressor-htt            Regression Modelling for HTT
  knnhtt                   KNN scaler modelling
  random-htt               Random Scaler Modelling
  cross-validate-logistic  Logistic Cross Validate the dataset
  cross-validate-random    Random Forest cross validate the dataset
  cross-vaidate-knn        KNN cross validate the dataset
  burn-class               Neural Burn classification
  xg-boost                 XGBoost classification
  help                     Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version

```

