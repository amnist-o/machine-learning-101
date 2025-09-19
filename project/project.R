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
# Plot histogram of price
df %>%
  ggplot(aes(x=price))+
  geom_histogram()
# Plot histogram of price by beds and property type
df %>%
  ggplot(aes(x=log1p(price)))+
  geom_histogram()+
  facet_grid(beds~property_type,scale="free")

# Splits data into training and testing sets
set.seed(777)
trainIndex <- createDataPartition(df$price, p = 0.5, list = FALSE, times = 1)

train_data <- df[trainIndex,]
validation_data  <- df[-trainIndex,]

# Scale the training and test data based on the training data mean and variance.
ScalingValues <- preProcess(train_data, method = c("center", "scale"))
train_data <- predict(ScalingValues, train_data)
validation_data <- predict(ScalingValues, validation_data)

# Control variables
numbers <- 10
repeats <- 20
rcvTunes <- 15 # tune number of models
seed <- 123
# repeated cross validation
rcvSeeds <- setSeeds(method = "repeatedcv", 
                     numbers = numbers, repeats = repeats, 
                     tunes = rcvTunes, seed = seed)


# Controls for the CV 
rcvControl <- trainControl(method = "repeatedcv", 
                           number = numbers, repeats = repeats,
                           seeds = rcvSeeds)

# Prepare the formula -----
## Preparing variables
outcome_var <- "price"
predictor_vars <- setdiff(names(df),outcome_var)
numeric_vars <- df %>% select(where(is.numeric)&!outcome_var) %>% names()
## Interaction terms
interaction_pairs <- combn(predictor_vars,2)
interaction_terms <- apply(interaction_pairs, 2, function(pair){
  paste(pair,collapse=":")
})
print(interaction_terms)
## Squared terms
squared_terms <- paste0("I(",numeric_vars,"^2)")
print(squared_terms)
## Full formula
formula_string <- paste(outcome_var,"~",
                        paste(c(predictor_vars,interaction_terms,squared_terms),
                              collapse = "+"))
full_formula <- as.formula(formula_string)
# Train data -----
set.seed(123)
lasso_fit <- train(full_formula,
                   data = train_data, 
                   method = "glmnet",
                   tuneGrid = expand.grid(alpha = 1, 
                                          # grid to search
                                          lambda = seq(from =0,
                                                       to=0.003,
                                                       length = 5)),
                   trControl = rcvControl)

ggplot(lasso_fit)   +
  ggtitle("Lasso Penalization") +
  labs(x = "Regularization parameter (Lambda)")+
  theme_minimal()

cbind(lasso_fit$bestTune$lambda)  %>% 
  kable(digits=3, col.names = c("lambda (exp)")) %>%
  kable_styling()

theme_models <-  theme_minimal()+ theme(plot.title = element_text(hjust = 0.5),
                                        legend.position = "none") 

lasso_varImp <- data.frame(variables = row.names(varImp(lasso_fit)$importance), varImp(lasso_fit)$importance)
# Below we set that we only show feature importance with a value larger than 3
# You can lower this if you want to see more variables, or increase it if you want to see fewer.
threshold = 2
lasso_varImp <- lasso_varImp[lasso_varImp$Overall > threshold,]
ggplot(data = lasso_varImp, mapping = aes(x=reorder(variables, Overall),
                                          y=Overall,
                                          fill=variables)) +
  coord_flip() + geom_bar(stat = "identity", position = "dodge") +
  theme_models +
  labs(x = "", y = "") +
  ggtitle("Feature Importance Lasso Regression") 


lasso_preds <- predict(lasso_fit, validation_data)
rmse(actual = validation_data$price, predicted = lasso_preds)