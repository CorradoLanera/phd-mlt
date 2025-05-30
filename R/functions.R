# R/functions.R
# Helper functions for the tidymodels classification workflow with targets

# -- Ensure necessary packages are loaded by targets in _targets.R --
# library(tidymodels)
# library(tidyverse)
# library(kknn)
# library(kernlab)
# library(rpart)
# library(ranger)
# library(datasets)
# library(future)

# ----------------------------------------------------------------------
# 1. DATA LOADING AND PREPARATION
# ----------------------------------------------------------------------

#' Load and Prepare Palmer Penguins Data
#'
#' Loads the native 'penguins' dataset, removes missing values,
#' and converts to a tibble.
#'
#' @return A tibble of the cleaned penguins dataset.
load_and_prep_data <- function() {
  data("penguins", package = "datasets")
  penguins |>
    remove_missing() |> # As per user's 03-tidymodels.R
    as_tibble()
}

#' Split Data into Training and Testing Sets
#'
#' Performs a stratified split of the data.
#'
#' @param data The input tibble.
#' @param prop Proportion of data to allocate to training.
#' @param strata Variable to stratify by.
#' @return An `rsplit` object.
split_data <- function(data, prop = 0.7, strata_var = "island") {
  initial_split(data, prop = prop, strata = {{ strata_var }})
}

# ----------------------------------------------------------------------
# 2. RECIPE DEFINITION
# ----------------------------------------------------------------------

#' Create a Classification Recipe
#'
#' Defines preprocessing steps: dummy coding, zero-variance filter,
#' and normalization.
#'
#' @param data Training data (tibble).
#' @param formula The model formula.
#' @return A `recipe` object.
create_classification_recipe <- function(
  data,
  formula_str = "island ~ bill_len + bill_dep + flipper_len + body_mass + sex"
) {
  recipe(stats::as.formula(formula_str), data = data) |>
    step_dummy(all_nominal_predictors()) |>
    step_zv(all_predictors()) |>
    step_normalize(all_numeric_predictors())
}

# ----------------------------------------------------------------------
# 3. MODEL SPECIFICATIONS (Parsnip)
# ----------------------------------------------------------------------

#' Define Model Specifications
#'
#' Creates parsnip model specifications for kNN, SVM (linear), Decision Tree, and Random Forest.
#' Hyperparameters to be tuned are marked with `tune()`.
#'
#' @param model_type A string indicating the model type: "knn", "svm_linear", "tree", "rf".
#' @return A `model_spec` object.
define_model_spec <- function(model_type) {
  switch(
    model_type,
    "knn" = nearest_neighbor(neighbors = tune()) |>
      set_engine("kknn") |>
      set_mode("classification"),
    "svm_linear" = svm_linear(cost = tune()) |>
      set_engine("kernlab") |> # kernlab for svmLinear, or LiblineaR
      set_mode("classification"),
    "tree" = decision_tree(cost_complexity = tune(), min_n = tune(), tree_depth = tune()) |>
      set_engine("rpart") |>
      set_mode("classification"),
    "rf" = rand_forest(mtry = tune(), trees = 3000, min_n = tune()) |> # trees set high as in 03-tidymodels.R
      set_engine("ranger", importance = "permutation") |>
      set_mode("classification"),
    stop("Unknown model_type: ", model_type)
  )
}

# ----------------------------------------------------------------------
# 4. WORKFLOWS
# ----------------------------------------------------------------------

#' Create a Tidymodels Workflow
#'
#' Bundles a recipe and a model specification.
#'
#' @param model_recipe A `recipe` object.
#' @param model_spec A `model_spec` object.
#' @return A `workflow` object.
create_tidymodels_workflow <- function(model_recipe, model_spec) {
  workflow() |>
    add_recipe(model_recipe) |>
    add_model(model_spec)
}

# ----------------------------------------------------------------------
# 5. RESAMPLING AND TUNING
# ----------------------------------------------------------------------

#' Define Cross-Validation Folds
#'
#' Creates k-fold cross-validation resamples.
#'
#' @param data Training data.
#' @param v Number of folds.
#' @param strata_var Variable to stratify by.
#' @return An `rset` object (e.g., `vfold_cv`).
define_cv_folds <- function(data, v = 5, strata_var = "island") {
  vfold_cv(data, v = v, strata = {{ strata_var }})
}

#' Define Classification Metrics Set
#'
#' Defines a set of metrics for evaluating classification models.
#'
#' @return A `metric_set` object.
define_classification_metrics <- function() {
  metric_set(accuracy, roc_auc, sensitivity, specificity)
}

#' Define Tuning Grids
#'
#' Creates hyperparameter grids for tuning.
#'
#' @param model_type A string indicating the model type.
#' @return A tibble representing the tuning grid.
define_tuning_grid <- function(model_type) {
  switch(
    model_type,
    "knn" = grid_regular(neighbors(range = c(1, 30)), levels = 30),
    "svm_linear" = grid_regular(cost(range = c(-3, 2)), levels = 10), # log10 scale
    "tree" = grid_regular(
      cost_complexity(range = c(-3, -1)), # log10 scale
      min_n(range = c(5, 40)),
      tree_depth(range = c(1, 10)),
      levels = 5 # 5x5x5 grid
    ),
    "rf" = grid_regular(
      mtry(range = c(1L, 5L)), # Max 5 predictors: bill_len, bill_dep, flipper_len, body_mass, (sex_male OR sex_female)
      min_n(range = c(2L, 20L)),
      levels = 5 # 5x5 grid
    ),
    stop("Unknown model_type for grid: ", model_type)
  )
}

#' Tune Model Grid
#'
#' Performs hyperparameter tuning using grid search.
#'
#' @param model_workflow The `workflow` object.
#' @param cv_folds The resampling `rset` object.
#' @param tuning_grid The hyperparameter grid.
#' @param metrics_set The `metric_set` object.
#' @return A tibble with tuning results.
tune_model_grid <- function(model_workflow, cv_folds, tuning_grid, metrics_set) {
  tune_grid(
    model_workflow,
    resamples = cv_folds,
    grid = tuning_grid,
    metrics = metrics_set,
    control = control_grid(save_pred = TRUE, verbose = TRUE) # save_pred for ROC etc.
  )
}

# ----------------------------------------------------------------------
# 6. MODEL SELECTION AND FINALIZATION
# ----------------------------------------------------------------------

#' Select Best Hyperparameters
#'
#' Selects the best hyperparameter combination based on a specified metric.
#'
#' @param tuned_results Results from `tune_grid()`.
#' @param metric The metric to optimize (e.g., "roc_auc").
#' @return A tibble with the best hyperparameter set.
select_best_hyperparams <- function(tuned_results, metric = "roc_auc") {
  select_best(tuned_results, metric = metric)
}

#' Finalize and Evaluate Model (Last Fit)
#'
#' Finalizes a workflow with the best hyperparameters and evaluates it
#' on the test set using the initial data split.
#'
#' @param model_workflow The base `workflow` object.
#' @param best_hyperparams A tibble with the best hyperparameters.
#' @param data_split The `rsplit` object from `initial_split()`.
#' @param metrics_set The `metric_set` to use for evaluation.
#' @return Results from `last_fit()`.
finalize_and_evaluate_model <- function(model_workflow, best_hyperparams, data_split, metrics_set) {
  final_workflow <- finalize_workflow(model_workflow, best_hyperparams)
  last_fit(final_workflow, split = data_split, metrics = metrics_set)
}

# ----------------------------------------------------------------------
# 7. RESULTS EXTRACTION AND PRESENTATION
# ----------------------------------------------------------------------

#' Collect and Summarize Tuning Results for Comparison
#'
#' Binds metrics from multiple tuned models for easy comparison.
#'
#' @param ... Named arguments where names are model identifiers (e.g., "kNN")
#'            and values are the corresponding tuning results from `tune_grid()`.
#' @return A tibble summarizing mean performance for each model and metric.
summarize_all_tuning_results <- function(...) {
  named_results <- list(...)
  if (length(named_results) == 0 || !all(sapply(names(named_results), nzchar))) {
    stop("All arguments must be named and there must be at least one.")
  }

  purrr::map2_dfr(named_results, names(named_results), ~{
    collect_metrics(.x) |>
      mutate(model = .y)
  })
}

#' Extract Final Model Metrics
#'
#' Extracts performance metrics from `last_fit()` results.
#'
#' @param last_fit_results Output from `last_fit()`.
#' @return A tibble of metrics.
extract_final_metrics <- function(last_fit_results) {
  collect_metrics(last_fit_results)
}

#' Extract Final Model Predictions
#'
#' Extracts predictions from `last_fit()` results.
#'
#' @param last_fit_results Output from `last_fit()`.
#' @return A tibble of predictions.
extract_final_predictions <- function(last_fit_results) {
  collect_predictions(last_fit_results)
}

#' Generate Confusion Matrix from Predictions
#'
#' @param final_predictions Tibble of predictions from `collect_predictions()`.
#' @param truth_var Name of the true outcome variable (string).
#' @param estimate_var Name of the predicted class variable (string, default ".pred_class").
#' @return A `conf_mat` object.
generate_confusion_matrix <- function(final_predictions, truth_var = "island", estimate_var = ".pred_class") {
  final_predictions |>
    conf_mat(truth = {{ truth_var }}, estimate = {{ estimate_var }})
}

#' Extract Trained Workflow from Last Fit
#'
#' @param last_fit_results Output from `last_fit()`.
#' @return The trained `workflow` object.
extract_trained_workflow <- function(last_fit_results) {
  extract_workflow(last_fit_results)
}
