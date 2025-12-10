# fMRI Foundation Model Datasets

## Dataset contributing guide

Every new dataset should have its own directory including:

- `README.md` giving background and step-by-step process for constructing the dataset.
- all scripts needed to curate, preprocess, and generate the final dataset(s).
- any metadata (provided it's publicly shareable) needed for running the scripts.

The provided scripts should be able to run *end-to-end* with as little manual intervention as possible. Any manual steps should be documented in the `README.md`. This might include:

- steps for gaining access and downloading source data.
- setting environment variables.
- setting hard-coded variables, e.g. local paths.
- getting access to destination data storage (e.g. R2, huggingface).

New datasets should follow roughly the same structure as existing datasets. See for example [`HCP-YA`](HCP-YA/). A good way to ensure your dataset is consistent is to copy and modify another recent dataset.
