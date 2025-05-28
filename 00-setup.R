install.packages("pak")

pak::pkg_install(c(
  # req dep
  c("cpp11", "progress"),
  # dev
  c("usethis"),
  # mlt
  c("kknn", "e1071", "rpart", "randomForest"),
  # eval
  c("pROC", "skimr"),
  # infrastructure
  c("tidyverse", "patchwork", "caret")
), dependencies = TRUE)

renv::snapshot()
