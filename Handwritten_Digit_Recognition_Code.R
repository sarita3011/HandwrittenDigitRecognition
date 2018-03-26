library(h2o)
library(caret)
library(data.table)
library(glmnet)
library(dplyr)

rm(list=ls())
setwd("~/MNIST")

##### read input ####
input <- fread("train.csv")
input <- as.data.frame(input)


#### make label a factor ####

input$label <- as.factor(input$label)

#######################################
# H2O implementation for NN ####
#######################################
# remove zero variance predictors
non_zero_var_col <- setdiff(colnames(input),colnames(input[,-1])[apply(input[,-1],2,var)==0])
input <- input %>% select(c(non_zero_var_col))

# split input to use 20% as validation in h2o
idx <- sample(1:nrow(input),0.8*nrow(input))
train <- input[idx,]
test <- input[-idx,] 

# initialization
localh2o <- h2o.init(nthreads = -1)

# create datasets for H2O
train_h2o <- as.h2o(train)
test_h2o <- as.h2o(test)

# hyper parameter to be given as input to h2o
hyper_params <- list(
  activation=c("Rectifier","Tanh","TanhWithDropout"),
  hidden=list(c(160,160),c(200,200),c(160,160,160),c(250,250),c(250,250,250),c(260,260),c(270,270),c(220,220),c(300,300),c(300)),
  l1=seq(0,1e-4,1e-6),
  l2=seq(0,1e-4,1e-6)
)
search_criteria = list(strategy = "RandomDiscrete", max_runtime_secs = 560,
                       max_models = 200, seed=12345, stopping_rounds=5, stopping_tolerance=1e-2)
dl_random_grid <- h2o.grid(
  algorithm="deeplearning",
  grid_id = "dl_grid_random",
  training_frame=train_h2o,
  validation_frame=test_h2o, 
  x=2:709, 
  y=1,
  epochs=30,
  stopping_metric="misclassification",
  stopping_tolerance=1e-5,        
  stopping_rounds=4,
  score_validation_samples=10000, ## downsample validation set for faster scoring
  score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
  max_w2=10,                      ## stability for Rectifier
  hyper_params = hyper_params,
  search_criteria = search_criteria
)                     

# model with lowest misclassification
grid <- h2o.getGrid("dl_grid_random",sort_by="err",decreasing=FALSE)
best_model <- h2o.getModel(grid@model_ids[[1]]) 
best_params <- best_model@allparameters
best_params$activation # best activation function
best_params$hidden # best number of units in hidden layers

# read test data for scoring
test_kaggle <- fread("test.csv",stringsAsFactors = FALSE)
test_kaggle <- as.data.frame(test_kaggle)
test_h2o <- as.h2o(test_kaggle)
pred <- h2o.predict(best_model,test_h2o)

################################
##### caret random forest
################################

# convert factor variable because 0-9 not valid column lables for R
levels(input$label) <- c("zero","one","two","three","four","five","six","seven","eight","nine")

# Create custom indices: myFolds
myFolds <- createFolds(input[,1], k = 5)

# Create reusable trainControl object: myControl
myControl <- trainControl(
  #  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE,
  savePredictions = TRUE,
  index = myFolds
)

# Fit random forest: model_rf
model_rf <- train(
  x = input[,-1], y = input[,1],
  metric = "ROC",
  method = "ranger",
  trControl = myControl,
  preProcess = c("zv","pca") # remover zero variance predictors and perform PCA
)

# read test data
test_kaggle <- fread("test.csv",stringsAsFactors = FALSE)
test_kaggle <- as.data.frame(test_kaggle)

# prediction for submission
pred <- predict(model_rf,test_kaggle)
levels(pred) <- c(0,1,2,3,4,5,6,7,8,9)
write.csv(pred,"first.csv")

