Interferometr
==============================

Repozytorium programu obslugujacego dane z interferometru

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- !!!! All data is not on repo!
    │   ├── raw            <- Data from third party sources.
    |   |    └── 1chanel
    |   |       ├── photo
    |   |       |   ├── training        <- real photo for test (from 00000.png to 19999.png)
    |   |       |   └── test            <- real photo training and validation (from 20000.png to 23999.png)
    |   |       └── reference
    |   ├── generated
    |   |   ├── noise
    |   |   |   └── photo               <- photo with noise only
    |   |   ├── noised
    |   |   |   ├── photo
    |   |   |   |   ├── training        <- generated photo with noise for training and validation (from 00000.png to 19999.png)
    |   |   |   |   └── test            <- generated photo with noise for test only (from 20000.png to 23999.png)
    |   |   |   └── reference
    |   |   |       ├── training        <- folder with epsilon.csv for noised training
    |   |   |       └── test            <- folder with epsilon.csv for noised test
    |   |   └── unnoised
    |   |       ├── photo
    |   |       |   ├── training        <- generated photo with noise for training only (from 00000.png to 06999.png)
    |   |       |   └── test            <- generated photo with noise for test and validation (from 07000.png to 09999.png)
    |   |       └── reference
    |   |           ├── training        <- folder with epsilon.csv for noised training
    |   |           └── test            <- folder with epsilon.csv for noised test
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── test_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
