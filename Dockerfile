# Dockerfile per containerizzare il progetto R con Tidymodels e RStudio Server

# ------------------------------------------------------------------------------
# 1. IMMAGINE BASE
# ------------------------------------------------------------------------------
# Usiamo rocker/verse, che include R, RStudio Server, Tidyverse, devtools, e Quarto CLI.
FROM rocker/verse:latest

# ------------------------------------------------------------------------------
# 2. METADATI (Opzionale)
# ------------------------------------------------------------------------------
LABEL maintainer="Corrado Lanera <corrado.lanera@ubep.unipd.it>"
LABEL description="Container Docker per progetto Tidymodels con RStudio, targets, e Quarto."

# ------------------------------------------------------------------------------
# 3. INSTALLAZIONE PACCHETTI R
# ------------------------------------------------------------------------------
# Passa a utente root per installare i pacchetti R a livello di sistema
# (o per installare dipendenze di sistema se necessario, anche se rocker/verse è molto completo).
USER root

# Installa i pacchetti R necessari per il progetto.
# È buona pratica specificare un repository CRAN.
RUN R -e "install.packages(c( \
    'targets', \
    'tarchetypes', \
    'tidymodels', \
    'palmerpenguins', \
    'kknn', \
    'ranger', \
    'kernlab', \
    'rpart', \
    'testthat', \
    'quarto', \
    'knitr', \
    'DT' \
), repos = 'https://cloud.r-project.org/', Ncpus = parallel::detectCores())"

# ------------------------------------------------------------------------------
# 4. CONFIGURAZIONE DELL'AMBIENTE DEL PROGETTO
# ------------------------------------------------------------------------------
# Crea una directory per il progetto all'interno della home dell'utente rstudio.
# L'utente rstudio (uid 1000) è l'utente standard in queste immagini rocker.
RUN mkdir -p /home/rstudio/project
WORKDIR /home/rstudio/project

# Copia i file del progetto dalla directory di build del Dockerfile
# nel working directory del container.
# Assicurati che questo Dockerfile sia nella root del tuo progetto R.
COPY functions.R .
COPY _targets.R .
COPY report.qmd .
# Se hai altri file o directory (es. una cartella 'data/'), copiali anche qui.
# Esempio: COPY data/ ./data/

# ------------------------------------------------------------------------------
# 5. PERMESSI
# ------------------------------------------------------------------------------
# Assicura che l'utente 'rstudio' sia proprietario dei file del progetto.
RUN chown -R rstudio:rstudio /home/rstudio/project

# ------------------------------------------------------------------------------
# 6. RIPRISTINA UTENTE
# ------------------------------------------------------------------------------
# Torna all'utente non-root 'rstudio' per l'esecuzione di RStudio Server.
USER rstudio


# ------------------------------------------------------------------------------
# ISTRUZIONI POST-BUILD (da leggere, non eseguite da Dockerfile)
# ------------------------------------------------------------------------------
# Per costruire l'immagine (esegui nella directory contenente il Dockerfile):
#   docker build -t phd_mlt .
#
# Per eseguire il container:
#   docker run -d --rm -p 8787:8787 -e PASSWORD=supersecret --name rstudio_analyses phd_mlt
#
# Per eseguire il container montando la directory del progetto locale (per sviluppo live):
#   docker run -d --rm -p 8787:8787 -e PASSWORD=supersecret -v "$(pwd)":/home/rstudio/project --name rstudio_dev phd_mlt
# ATTENZIONE: In questo caso, i file locali sovrascriveranno quelli copiati nell'immagine!!
#
# Apri il browser e vai a http://localhost:8787
# Username: rstudio
# Password: supersecret (o come configurato)
#
# All'interno di RStudio nel container, puoi eseguire:
#   targets::tar_make()
#   targets::tar_read(final_report) # Per ottenere il path del report generato
#   # Il report HTML sarà in /home/rstudio/project/report.html
# ------------------------------------------------------------------------------
