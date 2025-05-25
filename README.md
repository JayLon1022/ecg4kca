# ecg4kca

This is a repo for ecg4kca project.

## Table of Contents

- [Raw Data Download](#raw-data-download-skip-this-step-if-start-with-parquet)
- [Repo Structure](#repo-structure)
- [Contributing](#contributing)

## Raw Data Download (Skip this step if start with .parquet)

1. in `script/fetch_mimic_iv_ecg.sh`, change the `DEST_DIR` to your desired path

2. issue the following command to download the data

    ```bash
    chmod +x ./script/fetch_mimic_iv_ecg.sh
    ./script/fetch_mimic_iv_ecg.sh
    ```

## Repo Structure

This repo is organized as follows:

```bash
├── data/
├── notebook/
│   ├── 000.preprocess.ipynb    # raw data -> .parquet
│   ├── 001.main_CNN.ipynb      # train and evaluate CNN model
├── out/
├── script/                     # bash scripts
├── src/
│   ├── __init__.py
│   ├── helper.py               # helper classes
│   ├── models.py               # models TODO:
│   ├── utils.py                # generic utils
├── .gitignore
├── LICENSE
├── README.md                   # this file
```

## Contributing

1. Fork the repo
2. Create a new branch
3. Make your changes
4. Push to your branch
5. Create a pull request
