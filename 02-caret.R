#-----------------------------------------------------------------------
# R SCRIPT FOR MACHINE LEARNING WORKFLOW WITH CARET
#-----------------------------------------------------------------------
# This script addresses the same classification problem as in
# "01-raw-models.R" but utilizes the 'caret' framework.
#
# OBJECTIVE: The focus is not on the hyperparameters themselves, but on
# demonstrating how 'caret' provides a suitable infrastructure for
# automating tuning, comparing models, and reducing code complexity
# compared to manual management.
#
# ======================================================
# = OPTIMAL FOR QUICK PROTOTYPING AND MODEL COMPARISON =
# ======================================================
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
# 0. INITIAL SETUP
#-----------------------------------------------------------------------
set.seed(123)

library(caret)
library(tidyverse)

penguins_clean <- penguins |>
  remove_missing() |>
  as_tibble()

# CARET ADVANTAGE 1: STRATIFIED DATA SPLITTING
train_indices <- createDataPartition(
  penguins_clean$island,
  p = 0.7,
  list = FALSE
)
train_data <- penguins_clean[train_indices, ]
test_data  <- penguins_clean[-train_indices, ]


#-----------------------------------------------------------------------
# 1. THE CARET WORKFLOW: AUTOMATION AND ROBUSTNESS
#-----------------------------------------------------------------------

# CARET ADVANTAGE 2: CENTRALIZED TRAINING CONTROL (trainControl)
# Instead of writing for-loops and manually calculating metrics for each
# model, we define a training strategy *once*.
# Here, we specify 10-fold cross-validation.
# NB: `classProbs = TRUE` is necessary for calculating AUC.
train_control <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = multiClassSummary,
  verboseIter = TRUE
)

#-----------------------------------------------------------------------
# 2. HYPERPARAMETER TUNING WITH CARET
#-----------------------------------------------------------------------
# Here we see the biggest advantage: `caret` completely abstracts manual
# tuning loops. We only define the grid of hyperparameters to test,
# and the `train()` function handles everything else.



# --- 2.1 k-Nearest Neighbors (kNN) ---
tune_grid_knn <- expand.grid(
  kmax = 1:30,
  distance = 2, # kknn default
  kernel = "optimal" ## kknn default
)

# CARET ADVANTAGE 3: UNIFIED TRAINING FUNCTION AND PREPROCESSING
# The `train()` function has a consistent syntax for ALL models!!
#
#
# > SEE: https://topepo.github.io/caret/available-models.html
#
#
# It replaces the entire `knn_train` function and the `for` loop.
# - `method = "kknn"`: We specify the model.
# - `trControl`: We apply our CV strategy.
# - `tuneGrid`: We pass the hyperparameter grid.
# - `preProcess`: `caret` automatically handles scaling and centering
#   of predictors
knn_tuned_model <- train(
  island ~ bill_len + bill_dep + flipper_len + body_mass,
  data = train_data,
  method = "kknn",
  trControl = train_control,
  tuneGrid = tune_grid_knn,
  preProcess = c("center", "scale"),
  metric = "AUC"
)

# CARET ADVANTAGE 4: AUTOMATIC RESULTS AND PLOTS
# The `knn_tuned_model` object already contains all results.
knn_tuned_model

knn_tuned_model$bestTune
knn_tuned_model$finalModel

# The `plot()` command automatically generates tuning curves,
# replacing all manual ggplot code.
plot(
  knn_tuned_model,
  highlight = TRUE,
  main = "Tuning Curves for k-NN (via Caret)"
)

# --- 2.2 Support Vector Machine (SVM) ---

tune_grid_svm <- expand.grid(cost = 0.1 * 10^c(0:4))

svm_tuned_model <- train(
  island ~ bill_len + bill_dep + flipper_len + body_mass + sex,
  data = train_data,
  method = "svmLinear2",
  trControl = train_control,
  tuneGrid = tune_grid_svm,
  preProcess = c("center", "scale"),
  metric = "AUC"
)

svm_tuned_model
plot(
  svm_tuned_model,
  highlight = TRUE,
  main = "Tuning Curves for Radial SVM (via Caret)"
)


# --- 2.3 Decision Tree (rpart) ---

tune_grid_tree <- expand.grid(cp = c(0.001, 0.01, 0.05, 0.1, 0.2))

tree_tuned_model <- train(
  island ~ bill_len + bill_dep + flipper_len + body_mass + sex,
  data = train_data,
  method = "rpart",
  trControl = train_control,
  tuneGrid = tune_grid_tree,
  metric = "AUC"
)

tree_tuned_model
plot(
  tree_tuned_model,
  highlight = TRUE,
  main = "Tuning Curves for Decision Tree (via Caret)"
)


# --- 2.4 Random Forest ---

tune_grid_rf <- expand.grid(mtry = c(1, 2, 3, 4, 5))

# Note: The number of trees (ntree) is not a hyperparameter to "tune" in the
# classic sense, but should be set to a sufficiently high value. We pass it
# directly to the `train` function.
rf_tuned_model <- train(
  island ~ bill_len + bill_dep + flipper_len + body_mass + sex,
  data = train_data,
  method = "rf",
  trControl = train_control,
  tuneGrid = tune_grid_rf,
  metric = "AUC",
  ntree = 5000,
  importance = TRUE
)

rf_tuned_model
plot(
  rf_tuned_model,
  highlight = TRUE,
  main = "Tuning Curves for Random Forest (via Caret)"
)
plot(rf_tuned_model$finalModel)
ggplot(
  varImp(rf_tuned_model, scale = FALSE),
  main = "Variable Importance (Random Forest via Caret)"
)


#-----------------------------------------------------------------------
# 3. MODEL COMPARISON
#-----------------------------------------------------------------------

# CARET ADVANTAGE 5: DIRECT AND STATISTICALLY VALID COMPARISON
# the previous final comparison is based on a single test set!
# `caret` allows us to compare models based on their average performance
# during cross-validation, which is a more robust
# The `resamples` function collects CV results from all models.

model_list <- list(
  kNN = knn_tuned_model,
  SVM = svm_tuned_model,
  Tree = tree_tuned_model,
  RF = rf_tuned_model
)
resampling_results <- resamples(model_list)
summary(resampling_results)

# CARET ADVANTAGE 6: IMMEDIATE COMPARATIVE VISUALIZATION
# We can easily visualize performance distributions to
# compare models side-by-side.
resampling_results |>
  bwplot(
    metric = "AUC",
    main = "Model Comparison (AUC from Cross-Validation)"
  )
resampling_results |>
  bwplot(
    metric = "prAUC",
    main = "Model Comparison (AUC from Cross-Validation)"
  )

#-----------------------------------------------------------------------
# 4. FINAL EVALUATION ON THE TEST SET
#-----------------------------------------------------------------------

# CARET ADVANTAGE 7: SIMPLIFIED PREDICTION AND EVALUATION
# After choosing the best model, we can use it to make predictions
# on the test set.
#
# > The `predict` function is the same for all models.
final_predictions <- predict(rf_tuned_model, newdata = test_data)

# `confusionMatrix` provides a comprehensive evaluation report, much more
# detailed than the simple accuracy calculated manually.
final_evaluation <- confusionMatrix(
  final_predictions,
  reference = test_data$island
)
final_evaluation
