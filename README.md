# Transformational AI: Analyzing Ecommerce Large Datasets for Machine Learning

Udacity **AI Programming with Python** project focused on PCEP-aligned Python skills: virtual environments, data loading, pandas DataFrames, exception handling, lists/loops, and basic visualization.

Using subsets of the [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) Electronics datasets from Hugging Face, this notebook cleans product metadata, filters highly rated products, exports ML-ready files, and visualizes price vs. rating trends.

## Project overview

| Area | What you do |
|------|-------------|
| Environment | Set up Jupyter + install project dependencies |
| Reviews data | Stream the first 100 Electronics reviews into a DataFrame |
| Metadata | Stream the first 100 Electronics product metadata rows |
| Analysis | Inspect columns, dtypes, and DataFrame info; look up products with try/except |
| Cleaning | Keep products with a title, `average_rating >= 4.5`, and a valid price |
| Export | Save cleaned data as CSV and Parquet |
| Visualization | Histograms and scatterplots for highly rated products (`>= 4.7`) |

## Requirements

- Python 3.9+ (developed with Python 3.14)
- Jupyter Notebook (or Cursor / VS Code notebook support)

## Setup

```bash
# Clone the repo
git clone https://github.com/justinkumpe/data-driven-approach-to-ai-and-python_project.git
cd data-driven-approach-to-ai-and-python_project

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install "datasets==2.21.0" pandas matplotlib seaborn pyarrow notebook ipykernel

# Register the kernel (optional, for Cursor / VS Code)
python -m ipykernel install --user --name=udacity-pcep --display-name="Python (udacity venv)"
```

> **Note:** Pin `datasets==2.21.0`. Newer `datasets` releases (5.x) no longer support the Amazon Reviews 2023 loading script used in this assignment.

## Run the notebook

**Browser (Jupyter):**

```bash
source venv/bin/activate
jupyter notebook "Udacity - PCEP Project.ipynb"
```

**Cursor / VS Code:** open `Udacity - PCEP Project.ipynb`, select the **Python (udacity venv)** kernel, then **Run All**.

The first data-load cells stream from Hugging Face and may take a minute.

## Project structure

```text
.
├── Udacity - PCEP Project.ipynb   # Main assignment notebook
├── top_products.csv               # Cleaned products (rating ≥ 4.5, valid price)
├── top_products.parquet           # Same data in Parquet format
├── user_inputs.txt                # Reflection question responses
├── .gitignore
├── .vscode/settings.json          # Workspace interpreter settings
└── README.md
```

## Datasets

Both configs come from `McAuley-Lab/Amazon-Reviews-2023` (streamed, 100-row samples):

| Config | Role |
|--------|------|
| `raw_review_Electronics` | Customer reviews (rating, text, ASIN, etc.) |
| `raw_meta_Electronics` | Product metadata (title, average_rating, price, etc.) |

## Outputs

After a full run you should have:

- `top_products.csv` / `top_products.parquet` — filtered product set for the ML team
- `user_inputs.txt` — saved reflection answers and completion name

## Tech stack

- **pandas** — DataFrames and cleaning
- **datasets** — Hugging Face data loading (streaming)
- **matplotlib** / **seaborn** — histograms and scatterplots
- **pyarrow** — Parquet export

## License

Course project work for Udacity’s AI Programming with Python nanodegree / PCEP-aligned curriculum. Amazon Reviews 2023 data remains subject to the dataset authors’ terms on Hugging Face.
