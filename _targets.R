# _targets.R

library(crew) # parallel processing
library(targets)
library(tarchetypes)


# Global options
tar_option_set(
  packages = c(
    "tidymodels", "tidyverse", "datasets", "future",
    "kknn", "kernlab", "rpart", "ranger", # Model engines
    "vip", "knitr", "DT", "patchwork" # For report and variable importance
  ),
  error = "continue",
  # optional parallel processing
  controller = crew::crew_controller_local(
    name = "phd-mlt",
    workers = 4, # Adjust based on your system
  )
)

# Source functions from R/ directory
tar_source("R") # Assumes functions.R is in an R/ subdirectory

# Define the pipeline
list(
  # 1. Data Loading and Splitting
  tar_target(penguins_raw_data, load_and_prep_data()),
  tar_target(data_split, split_data(penguins_raw_data)),
  tar_target(train_df, training(data_split)),
  tar_target(test_df, testing(data_split)),

  # 2. Recipe and CV Folds
  tar_target(class_recipe, create_classification_recipe(train_df)),
  tar_target(cv_folds, define_cv_folds(train_df)),
  tar_target(class_metrics_set, define_classification_metrics()),

  # 3. Model Specifications and Tuning Grids
  # kNN
  tar_target(knn_model_spec, define_model_spec("knn")),
  tar_target(knn_tuning_grid, define_tuning_grid("knn")),
  tar_target(
    knn_workflow_obj,
    create_tidymodels_workflow(class_recipe, knn_model_spec)
  ),
  tar_target(
    knn_tuned_res,
    knn_workflow_obj |>
      tune_model_grid(cv_folds, knn_tuning_grid, class_metrics_set)
  ),

  # SVM Linear
  tar_target(svm_model_spec, define_model_spec("svm_linear")),
  tar_target(svm_tuning_grid, define_tuning_grid("svm_linear")),
  tar_target(
    svm_workflow_obj,
    create_tidymodels_workflow(class_recipe, svm_model_spec)
  ),
  tar_target(
    svm_tuned_res,
    svm_workflow_obj |>
      tune_model_grid(cv_folds, svm_tuning_grid, class_metrics_set)
  ),

  # Decision Tree
  tar_target(tree_model_spec, define_model_spec("tree")),
  tar_target(tree_tuning_grid, define_tuning_grid("tree")),
  tar_target(
    tree_workflow_obj,
    create_tidymodels_workflow(class_recipe, tree_model_spec)
  ),
  tar_target(
    tree_tuned_res,
    tree_workflow_obj |>
      tune_model_grid(cv_folds, tree_tuning_grid, class_metrics_set)
  ),

  # Random Forest
  tar_target(rf_model_spec, define_model_spec("rf")),
  tar_target(rf_tuning_grid, define_tuning_grid("rf")),
  tar_target(
    rf_workflow_obj,
    create_tidymodels_workflow(class_recipe, rf_model_spec)
  ),
  tar_target(
    rf_tuned_res,
    rf_workflow_obj |>
      tune_model_grid(cv_folds, rf_tuning_grid, class_metrics_set)
  ),

  # 4. Collect and Compare All Tuning Results
  tar_target(
    all_models_tuning_summary,
    summarize_all_tuning_results(
      kNN = knn_tuned_res,
      `SVM Linear` = svm_tuned_res,
      `Decision Tree` = tree_tuned_res,
      `Random Forest` = rf_tuned_res
    )
  ),
  tar_target(
    tuning_comparison_plot,
    all_models_tuning_summary |>
      filter(.metric == "roc_auc") |>
      ggplot(aes(model, mean, color = model)) +
      geom_point(size = 3) +
      geom_errorbar(
        aes(ymin = mean - std_err, ymax = mean + std_err),
        width = 0.2
      ) +
      labs(
        title = "Model Comparison (Mean ROC AUC from CV)",
        x = "Model Type",
        y = "Mean ROC AUC (+/- std_err)"
      ) +
      theme_minimal(base_size = 14) +
      theme(
        legend.position = "none",
        axis.text.x = element_text(angle = 45, hjust = 1)
      )
  ),

  # 5. Select Best Hyperparameters for a chosen model (e.g., Random Forest)
  #    and perform final fit.
  #    You could create targets for best params for *each* model if needed.
  tar_target(
    rf_best_params,
    select_best_hyperparams(rf_tuned_res, "roc_auc")
  ),
  tar_target(
    final_rf_evaluation,
    rf_workflow_obj |>
      finalize_and_evaluate_model(
        rf_best_params,
        data_split,
        class_metrics_set
      )
  ),
  tar_target(
    final_rf_metrics,
    extract_final_metrics(final_rf_evaluation)
  ),
  tar_target(
    final_rf_predictions_test,
    extract_final_predictions(final_rf_evaluation)
  ),
  tar_target(
    final_rf_conf_matrix,
    generate_confusion_matrix(final_rf_predictions_test)
  ),
  tar_target(
    final_rf_trained_workflow,
    extract_trained_workflow(final_rf_evaluation)
  ),
  tar_target(
    final_rf_vip_plot,
    {
      # Ensure the model fit object is extracted correctly for vip
      fit_obj <- extract_fit_parsnip(final_rf_trained_workflow)
      if (inherits(fit_obj$fit, "ranger")) {
        vip::vip(fit_obj) +
          labs(
            title = "Variable Importance - Final Random Forest Model"
          )
      } else {
        message("VIP plot not generated: final RF model fit is not of expected 'ranger' class.")
        NULL
      }
    }
  ),

  # 6. Report Generation
  tar_quarto(final_report, "report.qmd", quiet = FALSE)
)
