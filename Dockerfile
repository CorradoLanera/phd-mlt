# Dockerfile to containerize the R project with Tidymodels and RStudio Server

# ----------------------------------------------------------------------
# 1. BASE IMAGE
# ----------------------------------------------------------------------
# We use rocker/verse, which includes R, RStudio Server, Tidyverse, devtools, and Quarto CLI.
FROM rocker/verse:latest

# ----------------------------------------------------------------------
# 2. METADATA (Optional)
# ----------------------------------------------------------------------
LABEL "maintainer"="Corrado Lanera <corrado.lanera@ubep.unipd.it>"
LABEL "description"="Docker container for Tidymodels project with RStudio, targets, and Quarto."

# ----------------------------------------------------------------------
# 3. INSTALL R PACKAGES
# ----------------------------------------------------------------------

# Install the R packages needed for the project.
# It is good practice to specify a CRAN repository.
RUN R -e "install.packages('pak'); \
    pak::pkg_install(c( \
      # req dep \
      c('cpp11', 'progress', 'RcppEigen'), \
      # dev \
      c('usethis'), \
      # mlt \
      c('kknn', 'e1071', 'rpart', 'randomForest', 'ranger'), \
      # eval \
      c('pROC', 'skimr', 'vip'), \
      # infrastructure \
      c( \
        'tidyverse', 'patchwork', 'caret', 'tidymodels', \
        'targets', 'tarchetypes', 'crew', 'DT', 'patchwork' \
      ) \
    ), dependencies = TRUE)"

# ----------------------------------------------------------------------
# 4. CONFIGURING THE PROJECT ENVIRONMENT
# ----------------------------------------------------------------------
# Create a directory for the project within the rstudio user's home
# directory.
# The rstudio user (uid 1000) is the standard user in these rocker
# images.RUN mkdir -p /home/rstudio/project
WORKDIR /home/rstudio

# Copy the project files from the build directory of the Dockerfile
# into the working directory of the container.
# Make sure this Dockerfile is in the root of your R project.
COPY R/functions.R ./R/
COPY _targets.R .
COPY report.qmd .

# If you have other files or directories (e.g., a 'data/' folder), copy
# them here as well.
# Example: COPY data/ ./data/




# ----------------------------------------------------------------------
# POST-BUILD INSTRUCTIONS (to be read, not executed by Dockerfile)
# ----------------------------------------------------------------------
# To build the image (run in the directory containing the Dockerfile):
# docker build -t phd_mlt .
#
# To run the container:
# docker run -d --rm -p 8787:8787 -e PASSWORD=supersecret --name rstudio_analyses phd_mlt
#
# To run the container by mounting the local project directory
# (for live development):
# docker run -d --rm -p 8787:8787 -e PASSWORD=supersecret -v "$(pwd)":/home/rstudio/project --name rstudio_dev phd_mlt
# WARNING: In this case, the local files will overwrite those copied
# into the image!!!
#
# Open the browser and go to http://localhost:8787
# Username: rstudio
# Password: supersecret (or as configured)
#
# Inside RStudio in the container, you can execute:
# targets::tar_make()
# targets::tar_read(final_report) # To get the path to the generated
# report.
# # The HTML report will be in /home/rstudio/project/report.html.
#
# # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# # WARNING: pay attention that when you close the container,
#            EVERYTHIG WILL GONE, DEFINITELY!!
#            If you want to keep the results, you should mount a local
#            directory.
#
# To push on docker hub (you need an account):
# docker login
# docker tag phd_mlt <yourusername>/phd_mlt
# docker push <yourusername>/phd_mlt
#
# To pull the image from docker hub:
# docker pull <yourusername>/phd_mlt
#

# ----------------------------------------------------------------------
# I have already pushed it with my account (corradolanera).
# So, e.g., even if you do not have an account, and from any system with
# Docker installed and running, you already can run the image with:
#
# docker run -d --rm -p 8787:8787 -e PASSWORD=supersecret --name rstudio_analyses corradolanera/phd_mlt
#
# From localhost:8787, you can access the RStudio Server with the
# credential: usr: rstudio, pwd: supersecret
#
# From the RStudio Server, you can run the targets pipeline with:
# targets::tar_make()
#
# The final report will be generated in the project directory as
# report.html; and you can open it with the browser and download it!
#
# If you want, you can also modify some function in the R/functions.R
# file or the _targets.R file to adapt the pipeline and execution
# to your needs, and tar_make() the pipeline again, and generate a new
# report.
#
# ----------------------------------------------------------------------
