#########################################################
### Train a classification model with training images ###
#########################################################

### Author: Yuting Ma
### Project 3
### ADS Spring 2016


train <- function(dat_train, label_train, par=NULL){
  
  ### Train a Gradient Boosting Model (GBM) using processed features from training images
  
  ### Input: 
  ###  -  processed features from images 
  ###  -  class labels for training images
  ### Output: training model specification
  
  ### load libraries
  library("gbm")
  
  ### Train with gradient boosting model
  if(is.null(par)){
    depth <- 3
  } else {
    depth <- par$depth
  }
  
  fit_gbm <- gbm.fit(x=dat_train, y=label_train,
                     n.trees=1000,
                     distribution="bernoulli",
                     interaction.depth=depth, 
                     bag.fraction = 0.5,
                     verbose=FALSE)
  best_iter <- gbm.perf(fit_gbm, method="OOB", plot.it = FALSE)
  
  return(list(fit=fit_gbm, iter=best_iter))
}

rf_train=function(data, label,ntree,mtry,node)
{
  ### Train a Decision using processed features from training images
  
  ### Input: 
  ###  -  processed features from images 
  ###  -  class labels for training images
  ### Output: training model specification
  
  library(data.table)
  library(dplyr)
  library(randomForest)
  
  ### Train with decision model
  data=data.frame(data)
  data=mutate(data,label=factor(label))
  rf_fit <- randomForest(label~ .,
                         data=data,
                         importance=TRUE, 
                         ntree=ntree,
                         nodesize=node,
                         mtry=mtry)
  return(rf_fit)
}
