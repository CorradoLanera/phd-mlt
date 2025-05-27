#-----------------------------------------------------------------------
# R SCRIPT FOR MACHINE LEARNING MODEL COMPARISON ON PENGUINS DATASET.
#-----------------------------------------------------------------------
# This script demonstrates the application of kNN, SVM, Decision Trees,
# and Random Forest for classification and regression tasks on the
# 'penguins' dataset.
#
# Key hyperparameters are highlighted and performance compared.
# R version 4.5.0 or later (for the native penguins dataset).
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
# 0. INITIAL SETUP.
#-----------------------------------------------------------------------
set.seed(123)

library(kknn)  # kNN (classification)
library(e1071)  # SVM
library(rpart)  # Decision Trees
library(randomForest)  # Random Forest

library(pROC)  # AUC
library(skimr)  # Data exploration
library(tidyverse) # Data manipulation and visualization

penguins <- as_tibble(penguins)
skim(penguins)

penguins_clean <- remove_missing(penguins)
penguins_clean

ss <- nrow(penguins_clean)
sample_indices <- sample(seq_len(ss), size = 0.7 * ss)
train_data <- penguins_clean[sample_indices, ]
test_data <- penguins_clean[-sample_indices, ]

#-----------------------------------------------------------------------
# 1. Classify Penguin Island and Species.
#-----------------------------------------------------------------------

table(penguins_clean$island) / ss
table(train_data$island) / nrow(train_data)
table(test_data$island) / nrow(test_data)

#` Target: species (Adelie, Chinstrap, Gentoo)
#` Predictors: bill_len, bill_dep, flipper_len, body_mass, island, sex

calculate_accuracy <- function(predictions, actual) {
  mean(predictions == actual)
}

#` For kNN we need numerical predictors only.
#` We exclude 'year' as it is not a direct biological attribute
numeric_cols <- c("bill_len", "bill_dep", "flipper_len", "body_mass")

train_data_knn_x <- train_data |>
  select(all_of(numeric_cols))
train_data_knn_y <- train_data$island

test_data_knn_x <- test_data |>
  select(all_of(numeric_cols))
test_data_knn_y <- test_data$island


# --- 1.1 k-Nearest Neighbors (kNN) by Classification ---.

#` k-NN HYPERPARAMETER (kknn::kknn):
#  - (number of neighbors):
#`    - Description: Number of neighbor samples to be considered for
#       classification of a new point.
#`    - Impact of change:
#`        - k small: More flexible model, sensitive to noise, low bias, high variance (risk of overfitting).
#`        - large k: Smoother model, less sensitive to noise, high bias, low variance (risk of underfitting).
#`    - Modification steps: integers 1, 2, 3, ...


knn_pred <- kknn(
  island ~ .,
  train = bind_cols(train_data_knn_x, island = train_data_knn_y),
  test = bind_cols(test_data_knn_x, island = test_data_knn_y),
  k = 3
)

knn_accuracy <- calculate_accuracy(
  knn_pred$fitted.values,
  test_data_knn_y
)

knn_accuracy

knn_train <- function(k) {
  mod <- kknn(
    island ~ .,
    train = bind_cols(train_data_knn_x, island = train_data_knn_y),
    test = bind_cols(test_data_knn_x, island = test_data_knn_y),
    k = k
  )

  train_acc <- predict(mod, new.data = train_data_knn_x) |>
    calculate_accuracy(train_data_knn_y)
  test_acc <- fitted(mod) |>
    calculate_accuracy(test_data_knn_y)

  train_auc <- predict(mod, new.data = train_data_knn_x, type = "prob") |>
    multiclass.roc(train_data_knn_y, predictor = _) |>
    auc()
  test_auc <- predict(mod, new.data = test_data_knn_x, type = "prob") |>
    multiclass.roc(test_data_knn_y, predictor = _) |>
    auc()

  tibble(
    mod = rep("knn", 4),
    par_name = "k",
    par_value = k,
    stage = c("train", "test", "train", "test"),
    measure_name = c("accuracy", "accuracy", "auc", "auc"),
    measure_value = c(
      train_acc, test_acc, train_auc, test_auc
    )
  )
}

setup_classification_results <- function() {
  tibble(
    mod = character(),
    par_name = character(),
    par_value = numeric(),
    stage = character(),
    measure_name = character(),
    measure_value = numeric()
  )
}

knn_results <- setup_classification_results()
for (k in 1:30) {
  knn_results <- knn_results |>
    bind_rows(knn_train(k))
  usethis::ui_done("{k}-NN done.")
}

classification_results |>
  ggplot(aes(par_value, measure_value, color = stage)) +
  geom_line() +
  geom_point() +
  facet_wrap(~ measure_name) +
  labs(
    title = "Fine tuning curves for k-NN",
    subtitle = "higher means better",
    x = "k (Number of Neighbors)",
    y = "Performance Measure"
  ) +
  theme_minimal()



# --- 1.2 Support Vector Machine (SVM) per Classificazione ---

#` SVM HYPERPARAMETERS (e1071::svm):
#` - kernel: Type of kernel to be used.
#`    - Description: Transforms data into higher dimensional space to find a separating hyperplane.
#`    - Common values: "linear", "polynomial", "radial" (RBF), "sigmoid".
#`    - impact: "linear" is simple; "radial" is flexible and widely used; "polynomial" has 'degree' parameter.
#` - cost (C): Cost parameter of misclassification (regularization parameter).
#`    - description: Controls the trade-off between wide margin and correct classification of training points.
#`    - Impact:
#`        - C small: Wider margin, tolerates more misclassifications (underfitting, high bias).
#`        - Large C: Narrower margin, tries to classify all points correctly (overfitting, high variance).
#`    - Typical range: Logarithmic scale (e.g., 0.01, 0.1, 1, 10, 100).


svm_model <- svm(
  island ~ bill_len + bill_dep + flipper_len + body_mass + sex,
  data = train_data,
  probability = TRUE,
  kernel = "radial",
  cost = 1,
  gamma = 0.1
)

svm_pred <- predict(svm_model, test_data)
svm_accuracy <- calculate_accuracy(svm_pred_labels, test_data$island)


svm_probs <- predict(svm_model, test_data, probability = TRUE) |>
  attr("probabilities")
levels(test_data$island)
svm_probs |> str()
svm_probs <- svm_probs[, levels(test_data$island)]
svm_probs |> str()
levels(test_data$island)

svm_auc <- multiclass.roc(test_data$island, svm_probs) |>
  auc()

svm_train <- function(kernel, cost, gamma) {
  mod <- svm(
    island ~ bill_len + bill_dep + flipper_len + body_mass + sex,
    data = train_data,
    probability = TRUE,
    kernel = kernel,
    cost = cost,
    gamma = gamma
  )

  train_acc <- predict(mod, newdata = train_data) |>
    calculate_accuracy(train_data$island)
  test_acc <- predict(mod, newdata = test_data) |>
    calculate_accuracy(test_data$island)

  train_auc_prob <- predict(mod, newdata = train_data, probability = TRUE) |>
    attr("probabilities")
  train_auc_prob <- train_auc_prob[, levels(train_data$island)]
  train_auc <- train_auc_prob |>
    multiclass.roc(train_data$island, predictor = _) |>
    auc()

  test_auc <- predict(mod, newdata = test_data, probability = TRUE) |>
    attr("probabilities")
  test_auc <- test_auc[, levels(test_data$island)]
  test_auc <- test_auc |>
    multiclass.roc(test_data$island, predictor = _) |>
    auc()

  tibble(
    mod = "svm",
    kernel = kernel,
    cost = cost,
    gamma = gamma,
    stage = c("train", "test", "train", "test"),
    measure_name = c("accuracy", "accuracy", "auc", "auc"),
    measure_value = c(
      train_acc, test_acc, train_auc, test_auc
    )
  )
}

svm_class_res <- tibble(
    mod = character(),
    kernel = character(),
    cost = numeric(),
    gamma = numeric(),
    stage = character(),
    measure_name = character(),
    measure_value = numeric()
  )

i <- 1
for (kernel in c("linear", "polynomial", "radial")) {
  for (cost in c(0.1*10^c(0:4))) {
    for (gamma in c(0.01, 0.03, 0.1, 0.3, 1)) {
      svm_class_res <- svm_class_res |>
        bind_rows(svm_train(kernel, cost, gamma))
      usethis::ui_done(
        "[{i}/{3*5*5}] {kernel} SVM with cost={cost} and gamma={gamma} done."
      )
      i <- i + 1
    }
  }
}

svm_class_res |>
  nest(data = -c(mod, kernel)) |>
  mutate(data = set_names(data, kernel)) |>
  pull(data) |>
  imap(\(x, nm) {
    x |>
      ggplot(
        aes(cost, measure_value, colour = factor(gamma))
      ) +
      geom_point() +
      geom_line() +
      scale_x_log10() +
      coord_cartesian(ylim = c(0.45, 1)) +
      facet_grid(stage ~ measure_name) +
      labs(
        title = str_glue("Fine tuning curves for SVM - kernel {nm}"),
        subtitle = "higher means better",
        x = "Cost",
        y = "Measure"
      ) +
      theme_minimal()
  }) |>
  patchwork::wrap_plots(ncol = 3, guides = "collect")



# --- 1.3 Albero Decisionale (rpart) per Classificazione ---

#` DECISION TREE HYPERPARAMETERS (rpart::rpart):
#` - cp (Complexity Parameter): Complexity Parameter.
#`    - Description: Any split that does not decrease the overall error
#                    by a factor 'cp' is ignored. The main role of this
#                    parameter is to save computing time by pruning off
#                    splits that are obviously not worthwhile.
#                    Essentially, the user informs the program that any
#                    split which does not improve the fit by cp will
#                    likely be pruned off, and that hence the program
#                     need not pursue it.
#`    - Impact:
#`      - cp small: Larger, more complex tree (risk of overfitting).
#`      - cp large: Smaller and simpler tree (risk of underfitting).
#`    - Typical range: [0, 1], commonly small (e.g., 0.001, 0.01, 0.05).
#`    - Change step: Fine logarithmic or linear scale
#       (e.g. 0.001, 0.005, 0.01, ...).
#
#` - minsplit: Minimum number of observations in a node so that it can be split.
#`    - Description: Controls the minimum granularity to consider a split.
#`    - Impact:
#`      - minsplit small: Deeper and more complex trees.
#`      - minsplit large: Shallower and simpler trees.
#`    - Typical range: Depends on dataset size, e.g. 10, 20, 30. Default
#       is 20.
#`    - Modification steps: Integer.
#` - minbucket: Minimum number of observations in a leaf (terminal) node.
#`    - Description: Controls the minimum size of leaves. It is often minsplit/3.
#`    - Impact: Similar to minsplit, prevents too small and specific leaves.
#`    - Typical Range: Depends on minsplit, e.g. 5, 7, 10. Default is round(minsplit/3).
#`    - Modification steps: Integer.
#` - maxdepth: Maximum depth of the tree.
#`    - Description: Limits the growth of the tree in depth.
#`    - Impact: Greater depth = more complex model.
#`    - Typical range: Ex. 3-15, up to 30 (default rpart).
#`    - Modification steps: Integer.

tree_model <- rpart(
  island ~ bill_len + bill_dep + flipper_len + body_mass + sex,
  data = train_data,
  method = "class",
  cp = 0.01,        # complexity parameter (default 0.01);
  minsplit = 20,    # minimum number of observations that must exist in
                    #  a node in order for a split to be attempted
  minbucket = 7     # minimum number of observations in any terminal node
)
plot(tree_model, compress = TRUE)
text(tree_model, use.n = TRUE)
tree_pred_labels <- predict(tree_model, test_data, type = "class")
tree_pred_probs <- predict(tree_model, test_data, type = "prob")

tree_accuracy <- calculate_accuracy(tree_pred_labels, test_data$island)
tree_auc <- multiclass.roc(test_data$island, tree_pred_probs) |>
  auc()

tree_train <- function(cp, minsplit, minbucket) {
  mod <- rpart(
    island ~ bill_len + bill_dep + flipper_len + body_mass + sex,
    data = train_data,
    method = "class",
    cp = cp,
    minsplit = minsplit,
    minbucket = minbucket
  )
  train_acc <- predict(mod, newdata = train_data, type = "class") |>
    calculate_accuracy(train_data$island)
  test_acc <- predict(mod, newdata = test_data, type = "class") |>
    calculate_accuracy(test_data$island)

  train_probs <- predict(mod, newdata = train_data, type = "prob")
  train_probs <- train_probs[, levels(train_data$island)]
  train_auc <- multiclass.roc(train_data$island, train_probs) |>
    auc()
  test_probs <- predict(mod, newdata = test_data, type = "prob")
  test_probs <- test_probs[, levels(test_data$island)]
  test_auc <- multiclass.roc(test_data$island, test_probs) |>
    auc()

  tibble(
    mod = "tree",
    cp = cp,
    minsplit = minsplit,
    minbucket = minbucket,
    stage = c("train", "test", "train", "test"),
    measure_name = c("accuracy", "accuracy", "auc", "auc"),
    measure_value = c(
      train_acc, test_acc, train_auc, test_auc
    )
  )
}

tree_class_res <- tibble(
  mod = character(),
  cp = numeric(),
  minsplit = numeric(),
  minbucket = numeric(),
  stage = character(),
  measure_name = character(),
  measure_value = numeric()
)

i <- 1
for (cp in c(0.001, 0.01, 0.05, 0.1)) {
  for (minsplit in c(10, 20, 30)) {
    for (minbucket in c(3, 5, 7)) {
      tree_class_res <- tree_class_res |>
        bind_rows(tree_train(cp, minsplit, minbucket))
      usethis::ui_done(
        "[{i}/{4*3*3}] Tree with cp={cp}, minsplit={minsplit}, minbucket={minbucket} done."
      )
      i <- i + 1
    }
  }
}

tree_class_res |>
  ggplot(
    aes(cp, measure_value, color = factor(minsplit), shape = factor(minbucket), linetype = factor(minbucket))
  ) +
  geom_line() +
  geom_point() +
  facet_grid(stage ~ measure_name) +
  scale_x_log10() +
  labs(
    title = "Fine tuning curves for Decision Tree",
    subtitle = "higher means better",
    x = "Complexity Parameter (cp)",
    y = "Performance Measure"
  ) +
  theme_minimal()


# --- 1.4 Random Forest per Classificazione ---

#` RANDOM FOREST HYPERPARAMETERS (randomForest::randomForest):
#` - ntree: Number of trees to be built in the forest.
#`    - Description: Amount of trees in the ensemble.
#`    - Impact: Generally, more trees are better, up to a point where performance stabilizes. Increasing ntree does not lead to overfitting, but increases computation time.
#`    - Typical range: 100-1000 (or more). Default 500.
#`    - Change step: e.g. 100, 200, 500, 1000.
#` - mtry: Number of variables randomly selected at each split.
#`    - Description: Introduces randomness in feature selection for each tree by decorrelating them.
#`    - Impact:
#`      - mtry small: More randomness, more diverse trees, may reduce strength of individual trees but improve ensemble.
#`      - mtry large: Less randomness, more similar trees, approaches bagging.
#`    - Typical range: For classification, default is sqrt(number_predictors). For regression, default is number_predictors/3.
#`    - Change steps: Integer, around the default value.
#` - nodesize: Minimum size of terminal nodes (leaves).
#`    - Description: Similar to minbucket in rpart. Controls depth/complexity of individual trees.
#`    - Impact:
#`      - nodesize small: Deeper and more complex trees.
#`      - nodesize large: Shallower and simpler trees.
#`    - Typical range: For classification, default is 1. For regression, default is 5. Common values: 1, 5, 10.
#`    - Modification steps: Integer.
#` - maxnodes: Maximum number of terminal nodes that trees in the forest can have.
#`    - Description: Another way to control the size/complexity of trees.
#`    - Impact: If specified, interacts with nodesize.
#`    - Typical Range: Depends on the problem, e.g. 10, 20, ...
#`    - Modification steps: Integer.

rf_model <- randomForest(
  island ~ bill_len + bill_dep + flipper_len + body_mass + sex,
  data = train_data,
  xtest = test_data |>
    select(bill_len, bill_dep, flipper_len, body_mass, sex),
  ytest = test_data$island,
  ntree = 100,
  mtry = 2,
  importance = TRUE,
  keep.forest = TRUE
)
errors <- plot(rf_model)
as_tibble(errors)

rf_pred <- predict(rf_model, test_data, type = "response")
rf_accuracy <- calculate_accuracy(rf_pred, test_data$island)

rf_probs <- predict(rf_model, test_data, type = "prob")
rf_probs <- rf_probs[, levels(test_data$island)]
rf_auc <- multiclass.roc(test_data$island, rf_probs) |>
  auc()

rf_train <- function(mtry, ntree = 3000) {
  randomForest(
    island ~ bill_len + bill_dep + flipper_len + body_mass + sex,
    data = train_data,
    xtest = test_data |>
      select(bill_len, bill_dep, flipper_len, body_mass, sex),
    ytest = test_data$island,
    ntree = ntree,
    mtry = mtry,
    importance = TRUE,
    keep.forest = TRUE
  )
}

rf_class_res <- tibble(
  mod = character(),
  ntree = numeric(),
  mtry = numeric(),
  stage = character(),
  measure_name = character(),
  measure_value = numeric()
)

i <- 1
for (ntree in c(100, 300, 1000, 3000, 10000)) {
  for (mtry in c(1, 2, 3, 4, 5)) {
    rf_model <- rf_train(mtry, ntree)
    rf_pred <- predict(rf_model, test_data, type = "response")
    rf_accuracy <- calculate_accuracy(rf_pred, test_data$island)

    rf_probs <- predict(rf_model, test_data, type = "prob")
    rf_probs <- rf_probs[, levels(test_data$island)]
    rf_auc <- multiclass.roc(test_data$island, rf_probs) |>
      auc()

    rf_class_res <- rf_class_res |>
      bind_rows(tibble(
        mod = "rf",
        ntree = ntree,
        mtry = mtry,
        stage = c("train", "test", "train", "test"),
        measure_name = c("accuracy", "accuracy", "auc", "auc"),
        measure_value = c(
          rf_accuracy, rf_accuracy, rf_auc, rf_auc
        )
      ))

    usethis::ui_done(
      "[{i}/{5*5}] Random Forest with ntree={ntree} and mtry={mtry} done."
    )
    i <- i + 1
  }
}

rf_class_res |>
  ggplot(aes(ntree, measure_value, color = factor(mtry))) +
  geom_line() +
  geom_point() +
  facet_grid(stage ~ measure_name) +
  scale_x_log10() +
  labs(
    title = "Fine tuning curves for Random Forest",
    subtitle = "higher means better",
    x = "Number of Trees (ntree)",
    y = "Performance Measure"
  ) +
  theme_minimal()



## BEST MODELS FOR CLASSIFICATION

best_k_knn <- knn_results |>
  filter(
    stage == "test",
    measure_name == "auc"
  ) |>
  filter(measure_value == max(measure_value)) |>
  pull(par_value)
# add corresponding training measure
best_knn <- knn_results |>
  filter(par_value == best_k_knn)

best_par_svm <- svm_class_res |>
  filter(stage == "test", measure_name == "auc") |>
  arrange(desc(measure_value)) |>
  slice(1)

best_svm <- svm_class_res |>
  filter(
    kernel == best_par_svm$kernel,
    cost == best_par_svm$cost,
    gamma == best_par_svm$gamma
  )

best_par_tree <- tree_class_res |>
  filter(stage == "test", measure_name == "auc") |>
  arrange(desc(measure_value)) |>
  slice(1)

best_tree <- tree_class_res |>
  filter(
    cp == best_par_tree$cp,
    minsplit == best_par_tree$minsplit,
    minbucket == best_par_tree$minbucket
  )

best_par_rf <- rf_class_res |>
  filter(stage == "test", measure_name == "auc") |>
  arrange(desc(measure_value)) |>
  slice(1)

best_rf <- rf_class_res |>
  filter(
    ntree == best_par_rf$ntree,
    mtry == best_par_rf$mtry
  )

bests <- list(
  knn = best_knn,
  svm = best_svm,
  tree = best_tree,
  rf = best_rf
) |>
  bind_rows() |>
  select(mod, stage, measure_name, measure_value)

bests |>
  ggplot(aes(mod, measure_value, fill = stage)) +
  geom_col(position = "dodge") +
  facet_grid(measure_name ~ .) +
  labs(
    title = "Best Models for Classification",
    x = "Model",
    y = "Performance Measure"
  ) +
  theme_minimal() +
  scale_fill_brewer(palette = "Set1")
