# Setups -----
# My colors:
SIAP.color <- "#0385a8"
orange.color <- "#FF7F00"


#`r if(knitr:::pandoc_to() == "latex") {paste("\\large")}` 
# library -----
# Data management packages
library(tidyverse)
library(forcats)
library(modelsummary)

# Plotting packages
library(ggplot2)
library(RColorBrewer)
library(purrr)
library(rattle)
library(ggcorrplot)
library(grid)

# Model fitting packages
library(rpart)
library(caret)
library(leaps)  
library(ModelMetrics)

# Nice presentation of results
library(knitr)
library(papeR)
library(xtable)
library(kableExtra)

# parallel computing -----
# Sets up parallel computing for more efficient training
library(parallel)
nrcore <- detectCores()
cl <- parallel::makeCluster(nrcore-2, setup_strategy = "sequential")

library(doParallel)
registerDoParallel(cl)

# Seed setting -----
# This function is there to help you use parallel computing
# You do not need to change anything there nor to understand what's cooking in here
# function to set up random seeds
setSeeds <- function(method = "cv", 
                     numbers = 1, repeats = 1, 
                     tunes = NULL, seed = 123) 
{
  #B is the number of re-samples and integer vector 
  # of M (numbers + tune length if any)
  B <- if (method == "cv") numbers
  else if(method == "repeatedcv") numbers * repeats
  else NULL
  
  if(is.null(length)) {
    seeds <- NULL
  } else {
    set.seed(seed = seed)
    seeds <- vector(mode = "list", length = B)
    seeds <- 
      lapply(seeds, function(x) 
        sample.int(n = 1000000, 
                   size = numbers + ifelse(is.null(tunes), 
                                           0, tunes)))
    seeds[[length(seeds) + 1]] <- 
      sample.int(n = 1000000, size = 1)
  }
  # return seeds
}

# Load data -----
# Reading DHS survey data from SIAP's website
df <- read.csv("https://www.unsiap.or.jp/on_line/ML/M6-clean_data.csv")
str(df)

nrow(df)
ncol(df)

numeric <- select_if(df, is.numeric)
# We compute the correlation matrix of the covariates
corr_coef<-cor(numeric ,use = "p")
#And then plot it with nice options 
ggcorrplot(corr_coef, 
           type = "lower",         # lower triangle of the matrix only
           hc.order = TRUE,        # variable sorted from highest to lowest
           outline.col = "white",  #Color options
           lab = TRUE) + 
  ggtitle("Correlation between numerical variables")

df %>%
  ggplot(aes(x=review_scores_rating,y=price,colour = neighborhood))+
  geom_point()
