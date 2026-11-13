# Interpretable Machine Learning with SHAP

This project demonstrates how to explain sentiment-classification models with
SHAP-style feature attributions using the **Sentiment140** Twitter dataset.
It contains two executable Jupyter notebooks:

| Notebook | Model and explanation |
| --- | --- |
| [`TreeSHAP.ipynb`](./TreeSHAP.ipynb) | Trains an MLP and a Random Forest, then explains both models with a custom tree-surrogate SHAP implementation. |
| [`Kernel_SHAP.ipynb`](./Kernel_SHAP.ipynb) | Trains a Logistic Regression classifier with binary word features and explains it with `shap.KernelExplainer`. |

The notebooks produce model metrics, local explanations for individual
tweets, global feature-importance plots, and additivity/efficiency checks.
They are educational experiments, not a deployed sentiment-analysis service.

## Repository contents

```text
.
├── Kernel_SHAP.ipynb   # Kernel SHAP experiment
├── TreeSHAP.ipynb      # Tree-surrogate SHAP experiment
└── README.md
```

## Requirements

- Python 3.10 or newer is recommended
- Internet access on the first run, so `kagglehub` can download Sentiment140
- Jupyter Notebook/Lab, or Google Colab
- A few gigabytes of free disk space and several minutes for dataset loading,
  model training, and SHAP calculations

The notebooks use:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `scipy`
- `shap`
- `kagglehub`
- `jupyter`
- `ipykernel`

There is currently no `requirements.txt` or application entry point in this
repository. The installation command below is the complete dependency setup
for running the notebooks.

## Quick start with Google Colab

Colab is the simplest way to run the project because the notebooks already
contain Colab metadata and the dataset can be downloaded automatically.

1. Open one of these direct Colab links:
   - [Run TreeSHAP in Colab](https://colab.research.google.com/github/trishitasaha/IML_Project_Group5/blob/main/TreeSHAP.ipynb)
   - [Run Kernel SHAP in Colab](https://colab.research.google.com/github/trishitasaha/IML_Project_Group5/blob/main/Kernel_SHAP.ipynb)
2. You can also upload a notebook manually in
   [Google Colab](https://colab.research.google.com/).
3. Run every cell from top to bottom using **Runtime > Run all**.
4. If Colab asks for permission to install packages or access the internet,
   allow it.

`TreeSHAP.ipynb` installs its Python dependencies in its first code cell.
`Kernel_SHAP.ipynb` does not install packages automatically, so run this cell
before its other cells:

```python
%pip install -U kagglehub shap matplotlib scikit-learn pandas numpy scipy jupyter ipykernel
```

After installing packages, restart the runtime if Colab requests it, then
run the notebook from the first cell again.

## Local setup

### 1. Clone the repository

```bash
git clone https://github.com/trishitasaha/IML_Project_Group5.git
cd IML_Project_Group5
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

If PowerShell blocks activation, either allow it for the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

or run the notebooks with the environment's Python executable directly.

### 3. Install dependencies

Run this from the repository root while the virtual environment is active:

```bash
python -m pip install --upgrade pip
python -m pip install kagglehub shap matplotlib scikit-learn pandas numpy scipy jupyter ipykernel
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name iml-project-group5 --display-name "Python (IML Project Group 5)"
```

### 4. Start Jupyter

```bash
jupyter lab
```

Alternatively:

```bash
jupyter notebook
```

Open the notebook from the repository folder and select the
**Python (IML Project Group 5)** kernel.

## Dataset setup

Both notebooks use the public Kaggle dataset
[`kazanova/sentiment140`](https://www.kaggle.com/datasets/kazanova/sentiment140).
On a clean run, `kagglehub` downloads it automatically. The download code is
already present in both notebooks:

```python
import kagglehub
dataset_path = kagglehub.dataset_download("kazanova/sentiment140")
print(dataset_path)
```

The expected CSV is:

```text
training.1600000.processed.noemoticon.csv
```

The notebooks use the six-column, headerless Sentiment140 format:

```text
target, id, date, query/flag, user, text
```

The target values are `0` (negative) and `4` (positive). The notebooks map
these to binary labels `0` and `1`.

### Using a downloaded dataset instead

If automatic download is unavailable, download the dataset manually from
Kaggle and place the CSV in one of these locations:

- Repository root:
  `training.1600000.processed.noemoticon.csv`
- A `data` subfolder:
  `data/training.1600000.processed.noemoticon.csv`
- A Kaggle input folder (used by `Kernel_SHAP.ipynb`):
  `/kaggle/input/<dataset-folder>/training.1600000.processed.noemoticon.csv`

`Kernel_SHAP.ipynb` searches these locations before attempting a
`kagglehub` download. `TreeSHAP.ipynb` does **not** search the repository for
the CSV; it downloads the dataset with `kagglehub` and selects the first CSV
returned by that download. Therefore, for a fully offline TreeSHAP run, change
the dataset-loading cell to point to your local CSV before running it.

If Kaggle asks for authentication, configure Kaggle credentials according to
the [official Kaggle API instructions](https://www.kaggle.com/docs/api).
For a public dataset, manual placement of the CSV is an alternative.
Never commit Kaggle tokens or other credentials to this repository.

## Running `TreeSHAP.ipynb`

Run the cells in order. The notebook performs these stages:

1. Installs `kagglehub`, `shap`, `matplotlib`, `scikit-learn`, `pandas`, and
   `numpy` in the notebook's first code cell.
2. Downloads Sentiment140 and reads the CSV using `latin-1` encoding.
3. Samples 6,000 negative and 6,000 positive tweets.
4. Cleans tweets by lowercasing, removing URLs, mentions, punctuation, and
   repeated whitespace.
5. Builds a 120-feature TF-IDF representation.
6. Splits the data into 9,600 training and 2,400 test rows.
7. Trains:
   - an MLP classifier as a nonlinear black-box model;
   - a Random Forest classifier as a tree-based model.
8. Fits the custom `TreeSurrogateSHAP` explainer to each model's positive-class
   probability output.
9. Prints a local explanation for one test tweet and checks that
   `base value + sum(SHAP contributions)` matches the model prediction.
10. Displays local attribution plots and global mean-absolute-attribution plots
    for 100 test samples.

The custom explainer is implemented in the `TreeSurrogateSHAP` class inside
the notebook. It is included for demonstration and should not be treated as
a drop-in replacement for the official SHAP algorithms.

## Running `Kernel_SHAP.ipynb`

Run the cells in order. Install the dependencies first as described in the
Colab or local setup sections; this notebook intentionally has no install
cell. The notebook then performs these stages:

1. Locates or downloads the Sentiment140 CSV.
2. Reads the file in chunks, retaining a sample of each chunk.
3. Limits the working data to at most 20,000 examples per class.
4. Cleans URLs, mentions, and hashtags, removes very short tweets and
   duplicates, and creates binary word-presence features.
5. Keeps at most 250 vocabulary features and creates a stratified
   train/test split.
6. Trains a Logistic Regression sentiment classifier.
7. Reports accuracy, precision, recall, F1, ROC-AUC, a classification report,
   and a confusion matrix.
8. Selects up to eight balanced test examples and creates a ten-row SHAP
   background dataset.
9. Computes Kernel SHAP values with up to 200 sampled feature coalitions.
10. Checks SHAP additivity by comparing the model probability with
    `expected value + sum(SHAP values)`.
11. Displays waterfall plots for positive and negative examples, then global
    bar and beeswarm plots.

Kernel SHAP can be computationally expensive. For a faster first run, reduce
`N_EXPLAIN`, `NSAMPLES`, or `MAX_FEATURES` in the constants near the top of
the notebook. These changes affect runtime and may change the explanations.

## What results should look like?

Results can vary with package versions, hardware, and dataset sampling, so
the printed metrics are illustrative rather than fixed acceptance thresholds.
An earlier run produced approximately:

- `TreeSHAP.ipynb`: MLP accuracy `0.65`; Random Forest accuracy `0.65`
- `Kernel_SHAP.ipynb`: Logistic Regression accuracy `0.71`; ROC-AUC `0.78`
- Kernel SHAP reconstruction error: approximately `0.00000000` for the
  selected examples

For interpretation:

- A positive SHAP value pushes the predicted probability toward **positive**.
- A negative SHAP value pushes it toward **negative**.
- Mean absolute SHAP values measure average attribution magnitude, not whether
  a word is inherently positive or negative.
- The global rankings are based on the sampled examples and should not be
  treated as universal conclusions about sentiment.

## Troubleshooting

### `ModuleNotFoundError`

Make sure the notebook is using the virtual environment where dependencies
were installed. In a notebook, verify the interpreter with:

```python
import sys
print(sys.executable)
```

Then install into that exact interpreter:

```python
%pip install kagglehub shap matplotlib scikit-learn pandas numpy scipy
```

### Dataset download or authentication fails

Check your internet connection and Kaggle configuration. As a fallback,
download `training.1600000.processed.noemoticon.csv` manually and place it in
the repository root or `data` folder as described above.

### The notebook appears to hang

Model fitting and Kernel SHAP are the slowest stages. Wait for the current
cell to finish before starting another run. Reduce `NSAMPLES`,
`N_EXPLAIN`, or the dataset limits for a quick smoke test.

### A plot does not display

Run the plotting cell after the model and explanation cells, and use a
Jupyter kernel with an active Matplotlib backend. In Jupyter, this can help:

```python
%matplotlib inline
```

### Memory usage is high

`TreeSHAP.ipynb` converts its sparse TF-IDF matrix to a dense NumPy array.
Close other notebooks or reduce the sampled rows/features before converting.
`Kernel_SHAP.ipynb` also materializes sampled sparse matrices for SHAP input.

## Reproducibility

Both notebooks use a fixed seed of `42` for their main sampling and model
steps. Exact results can still differ across Python, scikit-learn, SHAP, and
hardware versions. The notebooks intentionally do not pin dependency versions;
for a fully frozen research environment, record the installed versions after
a successful run with:

```bash
python -m pip freeze > requirements-lock.txt
```

The generated lock file is local environment information and is not required
to run the repository.
