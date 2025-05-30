# Penguin Machine Learning Workflow

This repository demonstrates multiple approaches for building, tuning and comparing classification models on the Palmer Penguins dataset. The goal is to predict penguin islands using different machine learning libraries in R and to orchestrate the workflow with the `targets` package.

## Repository Structure

- `00-setup.R` – installs required packages and initializes the project.
- `01-raw-models.R` – manual implementation of kNN, SVM, decision tree and random forest models.
- `02-caret.R` – the same analysis using the **caret** framework.
- `03-tidymodels.R` – a tidymodels-based workflow.
- `R/functions.R` – helper functions used in the `targets` pipeline.
- `_targets.R` – defines the pipeline to automate preprocessing, tuning and reporting.
- `report.qmd` – Quarto document used to generate `report.html` after the pipeline completes.
- `Dockerfile` – optional container with RStudio and all dependencies preinstalled.

## Getting Started

1. **Install R (\>= 4.5) and [renv](https://rstudio.github.io/renv/)**.
   ```R
   renv::restore()
   ```

2. **Run the pipeline**
   ```R
   targets::tar_make()
   ```
   The final report will be produced as `report.html` in the project root.

3. **Optional: Use Docker**
   Build and run the container to launch RStudio Server with all packages installed:
   ```bash
   docker run -d --rm -p 8787:8787 -e PASSWORD=supersecret corradolanera/phd_mlt
   ```
   Log in at `http://localhost:8787` (user `rstudio`, password `supersecret`) and execute `targets::tar_make()` inside RStudio.

## License

This project is provided for educational purposes. See the source files for authorship information.
