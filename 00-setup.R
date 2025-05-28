install.packages("pak")

pak::pkg_install(c(
  # req dep
  c("cpp11", "progress", "RcppEigen"),
  # dev
  c("usethis"),
  # mlt
  c("kknn", "e1071", "rpart", "randomForest", "ranger"),
  # eval
  c("pROC", "skimr", "vip"),
  # infrastructure
  c("tidyverse", "patchwork", "caret", "tidymodels")
), dependencies = TRUE)

renv::snapshot()
