#-----------------------------------------------------------------------
# R SCRIPT FOR MACHINE LEARNING WORKFLOW WITH TIDYMODELS
#-----------------------------------------------------------------------
# This script addresses the same classification problem as in
# "02-caret.R" but utilizes the 'tidymodels' framework.
#
# OBJECTIVE: To demonstrate how 'tidymodels' provides a robust,
# uniform, and modular infrastructure. This facilitates code reuse,
# updates, and maintenance, making it ideal for projects that
# extend beyond a single coding session.
#
# =====================================================================
# = OPTIMAL FOR STRUCTURED, REPRODUCIBLE, AND MAINTAINABLE PROJECTS =
# =====================================================================
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
# 0. INITIAL SETUP
#-----------------------------------------------------------------------
# Tidymodels is a collection of packages, each serving a specific purpose
# in the modeling pipeline.
set.seed(123)

library(tidymodels)
library(tidyverse)

# optional parallel processing
library(future)
plan(multisession, workers = 2)

# Engine packages (tidymodels separates model specification from the engine)
library(kknn)
library(kernlab)  # For SVM
library(rpart)
library(ranger)        # For Random Forest (a fast implementation)

# Load and clean data (consistent with the caret script)
data("penguins", package = "datasets")
penguins_clean <- penguins |>
  remove_missing() |> # Handles missing values
  as_tibble()

# TIDYMODELS ADVANTAGE 1: ROBUST DATA SPLITTING (rsample package)
# `initial_split` performs stratified sampling by default on the
#   outcome variable when it's a factor, ensuring class proportions are maintained.
penguins_split <- penguins_clean |>
  initial_split(prop = 0.7, strata = island)
train_data <- training(penguins_split)
test_data  <- testing(penguins_split)

#-----------------------------------------------------------------------
# 1. THE TIDYMODELS WORKFLOW: MODULARITY AND UNIFORMITY
#-----------------------------------------------------------------------

# TIDYMODELS ADVANTAGE 2: PREPROCESSING WITH RECIPES (recipes package)
# A 'recipe' defines a sequence of preprocessing steps. It's highly modular
# and reusable. Steps are defined *before* training and are applied
# consistently during resampling and to new data.
#
# > This prevents data leakage!!
#
classification_recipe <- recipe(
  train_data,
  island ~ bill_len + bill_dep + flipper_len + body_mass + sex
) |>
  # Convert factor predictors to dummy variables
  step_dummy(all_nominal_predictors()) |>
  # Remove zero-variance predictors
  step_zv(all_predictors()) |>
  # Center and scale numeric predictors
  step_normalize(all_numeric_predictors())

# You can `prep()` and `bake()`:
prep_recipe <- prep(classification_recipe)
prep_recipe
bake(prep_recipe, new_data = NULL)

# TIDYMODELS ADVANTAGE 3: UNIFIED MODEL SPECIFICATION (parsnip package)
# `parsnip` provides a tidy, unified interface to specify models,
# separating the model *type* (e.g., kNN, SVM) from its computational *engine*
# (e.g., kknn, e1071) and its *mode* (classification/regression).
# This makes it easy to swap engines or try different models with minimal code changes.
#
# > show_engines("svm_linear")
# > https://www.tidymodels.org/find/parsnip/
#
# Model specifications
#  - hyperparameters that will be tuned are marked with `tune()`
knn_spec <- nearest_neighbor(neighbors = tune()) |>
  set_engine("kknn") |>
  set_mode("classification")

# svm_rbf(cost = tune(), rbf_sigma = tune()) for radial
svm_spec <- svm_linear(cost = tune()) |>
  set_engine("kernlab") |>
  set_mode("classification")

tree_spec <- decision_tree(
    cost_complexity = tune(),
    min_n = tune(),
    tree_depth = tune()
  ) |>
  set_engine("rpart") |>
  set_mode("classification")

rf_spec <- rand_forest(mtry = tune(), trees = 5000, min_n = tune()) |>
  set_engine("ranger", importance = "permutation") |>
  set_mode("classification")

# TIDYMODELS ADVANTAGE 4: WORKFLOWS (workflows package)
# A 'workflow' bundles a preprocessor (like a recipe) and a model specification.
# This is a central object that makes it easy to manage the modeling pipeline.
knn_workflow  <- workflow() |>
  add_recipe(classification_recipe) |>
  add_model(knn_spec)
svm_workflow  <- workflow() |>
  add_recipe(classification_recipe) |>
  add_model(svm_spec)
tree_workflow <- workflow() |>
  add_recipe(classification_recipe) |>
  add_model(tree_spec)
rf_workflow   <- workflow() |>
  add_recipe(classification_recipe) |>
  add_model(rf_spec)

#-----------------------------------------------------------------------
# 2. HYPERPARAMETER TUNING WITH TIDYMODELS (tune and dials packages)
#-----------------------------------------------------------------------
# `tune` provides functions for hyperparameter tuning using resampling.
# `dials` helps in creating and managing tuning parameter ranges.

# TIDYMODELS ADVANTAGE 5: ROBUST RESAMPLING FOR TUNING (rsample package)
# We define our resampling strategy (e.g., cross-validation) once.
set.seed(456) # Different seed for resampling
cv_folds <- vfold_cv(train_data, v = 5, strata = island)
cv_folds

# Define metrics to evaluate
# `metric_set` is used to specify which metrics to compute.
class_metrics <- metric_set(accuracy, roc_auc, sensitivity, specificity)

# --- 2.1 k-Nearest Neighbors (kNN) Tuning ---
knn_grid <- grid_regular(
  neighbors(range = c(1, 100)),
  levels = 100
)
knn_tuned_results <- knn_workflow |>
  tune_grid(
    resamples = cv_folds,
    grid = knn_grid,
    metrics = class_metrics,
    control = control_grid(save_pred = TRUE, verbose = TRUE)
  )

# TIDYMODELS ADVANTAGE 7: EXPLORING TUNING RESULTS (tune and ggplot2)
# `collect_metrics()` gathers performance metrics across all tuning parameters.
# `autoplot()` can often provide quick visualizations.
collect_metrics(knn_tuned_results)
autoplot(knn_tuned_results) +
  labs(title = "Tuning Curves for k-NN (via Tidymodels)")
show_best(knn_tuned_results, metric = "roc_auc", n = 5)


# --- 2.2 Support Vector Machine (SVM) Linear Tuning ---
svm_grid <- grid_regular(cost(range = c(-6, 10)), levels = 10)

svm_tuned_results <- tune_grid(
  svm_workflow,
  resamples = cv_folds,
  grid = svm_grid,
  metrics = class_metrics,
  control = control_grid(verbose = TRUE)
)

collect_metrics(svm_tuned_results)
autoplot(svm_tuned_results) +
  labs(title = "Tuning Curves for Linear SVM (via Tidymodels)")
show_best(svm_tuned_results, metric = "roc_auc", n = 5)


# --- 2.3 Decision Tree (rpart) Tuning ---
tree_grid <- grid_regular(
  cost_complexity(range = c(-3, -1)),
  min_n(range = c(5, 40)),
  tree_depth(range = c(1, 10)),
  levels = 5
)

tree_tuned_results <- tune_grid(
  tree_workflow,
  resamples = cv_folds,
  grid = tree_grid,
  metrics = class_metrics,
  control = control_grid(verbose = TRUE)
)

collect_metrics(tree_tuned_results)
autoplot(tree_tuned_results) +
  labs(title = "Tuning Curves for Decision Tree (via Tidymodels)")
show_best(tree_tuned_results, metric = "roc_auc", n = 5)


# --- 2.4 Random Forest (ranger) Tuning ---
rf_grid <- grid_regular(
  mtry(range = c(1L, 5L)),
  min_n(range = c(2L, 20L)),
  levels = 5
)

rf_tuned_results <- tune_grid(
  rf_workflow,
  resamples = cv_folds,
  grid = rf_grid,
  metrics = class_metrics,
  control = control_grid(verbose = TRUE)
)

collect_metrics(rf_tuned_results)
autoplot(rf_tuned_results) +
  labs(title = "Tuning Curves for Random Forest (via Tidymodels)")
show_best(rf_tuned_results, metric = "roc_auc", n = 5)


#-----------------------------------------------------------------------
# 3. COMPARING MODELS AND SELECTING THE BEST
#-----------------------------------------------------------------------
# TIDYMODELS ADVANTAGE 8: UNIFIED COLLECTION AND COMPARISON OF RESULTS
# We can collect all metrics and compare models systematically.
all_tuning_results <- bind_rows(
  collect_metrics(knn_tuned_results) |>
    mutate(model = "kNN"),
  collect_metrics(svm_tuned_results) |>
    mutate(model = "SVM Linear"),
  collect_metrics(tree_tuned_results) |>
    mutate(model = "Decision Tree"),
  collect_metrics(rf_tuned_results) |>
    mutate(model = "Random Forest")
)

# Plotting comparison (example for roc_auc)
all_tuning_results |>
  filter(.metric == "roc_auc") |>
  ggplot(aes(x = model, y = mean, color = model)) +
  geom_point(size = 3) +
  geom_errorbar(
    aes(ymin = mean - std_err, ymax = mean + std_err),
    width = 0.2) +
  labs(title = "Model Comparison (Mean ROC AUC from Cross-Validation)",
       x = "Model Type",
       y = "Mean ROC AUC (+/- std_err)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")


# Select the best hyperparameters for each model type
best_knn_params  <- select_best(knn_tuned_results, metric = "roc_auc")
best_svm_params  <- select_best(svm_tuned_results, metric = "roc_auc")
best_tree_params <- select_best(tree_tuned_results, metric = "roc_auc")
best_rf_params   <- select_best(rf_tuned_results, metric = "roc_auc")

#-----------------------------------------------------------------------
# 4. FINALIZING THE MODEL AND EVALUATING ON THE TEST SET
#-----------------------------------------------------------------------
# TIDYMODELS ADVANTAGE 9: FINALIZING WORKFLOWS AND `last_fit()`
# `finalize_workflow()` updates a workflow with the best hyperparameters.
# `last_fit()` fits this finalized workflow on the full training set
# and evaluates it on the test set. This is a streamlined process.
#
# Let's assume Random Forest was the best overall, or we want to evaluate it.
final_rf_workflow <- finalize_workflow(rf_workflow, best_rf_params)
final_rf_workflow

final_rf_fit_results <- last_fit(
  final_rf_workflow,
  split = penguins_split,
  metrics = class_metrics
)

collect_metrics(final_rf_fit_results)
final_rf_predictions <- collect_predictions(final_rf_fit_results)
final_rf_predictions

conf_mat_rf <- final_rf_predictions |>
  conf_mat(truth = island, estimate = .pred_class)
conf_mat_rf
autoplot(conf_mat_rf, type = "heatmap") +
  labs(title = "Confusion Matrix for Final RF Model (Test Set)")

final_trained_rf_workflow <- extract_workflow(final_rf_fit_results)
final_trained_rf_workflow |> predict(new_data = test_data |> slice(1:5))

# Variable importance from the final model
final_trained_rf_workflow |>
  extract_fit_parsnip() |>
  vip::vip()

