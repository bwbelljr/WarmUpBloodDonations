# Clear workspace
rm(list=ls())

# Read training data from csv into R dataframe
train_df <- read.csv("Warm_Up_Predict_Blood_Donations_-_Traning_Data.csv")

names(train_df)

summary(train_df)

# Here are my variables:
# X (participant ID)
# Months.since.Last.Donation (integer-valued)
# Number.of.Donations  (integer-valued)
# Total.Volume.Donated..c.c..  (integer-valued)
# Months.since.First.Donation  (integer-valued)
# Made.Donation.in.March.2007  (integer-valued)

# Confirming that I can use the above variable names within data frame
summary(train_df$X)
summary(train_df$Months.since.Last.Donation)
summary(train_df$Number.of.Donations)
summary(train_df$Total.Volume.Donated..c.c..)
summary(train_df$Months.since.First.Donation)
summary(train_df$Made.Donation.in.March.2007)

# Checking class/type of each variable
class(train_df$X)
class(train_df$Months.since.Last.Donation)
class(train_df$Number.of.Donations)
class(train_df$Total.Volume.Donated..c.c..)
class(train_df$Months.since.First.Donation)
class(train_df$Made.Donation.in.March.2007)

# Feature Engineering
# First I want to apply preliminary transformations to the data
# Then, I can apply additional transformations - spatial sign as well as Box Cox
# From there, I will pass it off to SuperLearner and allow it screening algprithms
# to decide which variables to include

# Simple Transformations

# Create new variable for how often a participant donates (in months)
train_df$num_months_per_donation <- (train_df$Months.since.First.Donation - train_df$Months.since.Last.Donation) / train_df$Number.of.Donations

# Create new variable to estimate how much blood volume is donated per number of donations
train_df$volume_per_donation <- (train_df$Total.Volume.Donated..c.c..)/(train_df$Number.of.Donations)

# Create new variable to estimate how much blood volume is donated per number of months
train_df$volume_per_donation <- (train_df$Total.Volume.Donated..c.c..)/(train_df$Months.since.First.Donation - train_df$Months.since.Last.Donation + 1)

# Let's create Log base 10 transformations of the predictor variables
train_df$Months.since.Last.Donation_log10 <- log10(train_df$Months.since.Last.Donation)
train_df$Number.of.Donations_log10 <- log10(train_df$Number.of.Donations)
train_df$Total.Volume.Donated..c.c.._log10 <- log10(train_df$Total.Volume.Donated..c.c..)
train_df$Months.since.First.Donation_log10 <- log10(train_df$Months.since.First.Donation)
train_df$num_months_per_donation_log10 <- log10(train_df$num_months_per_donation)
train_df$volume_per_donation_log10 <- log10(train_df$volume_per_donation)

# Let's create Log base 2 transformations of the predictor variables
train_df$Months.since.Last.Donation_log2 <- log2(train_df$Months.since.Last.Donation)
train_df$Number.of.Donations_log2 <- log2(train_df$Number.of.Donations)
train_df$Total.Volume.Donated..c.c.._log2 <- log2(train_df$Total.Volume.Donated..c.c..)
train_df$Months.since.First.Donation_log2 <- log2(train_df$Months.since.First.Donation)
train_df$num_months_per_donation_log2 <- log2(train_df$num_months_per_donation)
train_df$volume_per_donation_log2 <- log2(train_df$volume_per_donation)

# Let's create exponential transformations of the predictor variables
train_df$Months.since.Last.Donation_exp <- exp(train_df$Months.since.Last.Donation)
train_df$Number.of.Donations_exp <- exp(train_df$Number.of.Donations)
train_df$Total.Volume.Donated..c.c.._exp <- exp(train_df$Total.Volume.Donated..c.c..)
train_df$Months.since.First.Donation_exp <- exp(train_df$Months.since.First.Donation)
train_df$num_months_per_donation_exp <- exp(train_df$num_months_per_donation)
train_df$volume_per_donation_exp <- exp(train_df$volume_per_donation)

# Let's create sine transformations of the predictor variables
train_df$Months.since.Last.Donation_sin <- sin(train_df$Months.since.Last.Donation)
train_df$Number.of.Donations_sin <- sin(train_df$Number.of.Donations)
train_df$Total.Volume.Donated..c.c.._sin <- sin(train_df$Total.Volume.Donated..c.c..)
train_df$Months.since.First.Donation_sin <- sin(train_df$Months.since.First.Donation)
train_df$num_months_per_donation_sin <- sin(train_df$num_months_per_donation)
train_df$volume_per_donation_sin <- sin(train_df$volume_per_donation)

# Let's create cosine transformations of the predictor variables
train_df$Months.since.Last.Donation_cos <- cos(train_df$Months.since.Last.Donation)
train_df$Number.of.Donations_cos <- cos(train_df$Number.of.Donations)
train_df$Total.Volume.Donated..c.c.._cos <- cos(train_df$Total.Volume.Donated..c.c..)
train_df$Months.since.First.Donation_cos <- cos(train_df$Months.since.First.Donation)
train_df$num_months_per_donation_cos <- cos(train_df$num_months_per_donation)
train_df$volume_per_donation_cos <- cos(train_df$volume_per_donation)

# Let's create tangent transformations of the predictor variables
train_df$Months.since.Last.Donation_tan <- tan(train_df$Months.since.Last.Donation)
train_df$Number.of.Donations_tan <- tan(train_df$Number.of.Donations)
train_df$Total.Volume.Donated..c.c.._tan <- tan(train_df$Total.Volume.Donated..c.c..)
train_df$Months.since.First.Donation_tan <- tan(train_df$Months.since.First.Donation)
train_df$num_months_per_donation_tan <- tan(train_df$num_months_per_donation)
train_df$volume_per_donation_tan <- tan(train_df$volume_per_donation)

# Let's create Squared transformations of the predictor variables
train_df$Months.since.Last.Donation_sq <- (train_df$Months.since.Last.Donation)^2
train_df$Number.of.Donations_sq <- (train_df$Number.of.Donations)^2
train_df$Total.Volume.Donated..c.c.._sq <- (train_df$Total.Volume.Donated..c.c..)^2
train_df$Months.since.First.Donation_sq <- (train_df$Months.since.First.Donation)^2
train_df$num_months_per_donation_sq <- (train_df$num_months_per_donation)^2
train_df$volume_per_donation_sq <- (train_df$volume_per_donation)^2

# Let's create Cubic transformations of the predictor variables
train_df$Months.since.Last.Donation_cub <- (train_df$Months.since.Last.Donation)^3
train_df$Number.of.Donations_cub <- (train_df$Number.of.Donations)^3
train_df$Total.Volume.Donated..c.c.._cub <- (train_df$Total.Volume.Donated..c.c..)^3
train_df$Months.since.First.Donation_cub <- (train_df$Months.since.First.Donation)^3
train_df$num_months_per_donation_cub <- (train_df$num_months_per_donation)^3
train_df$volume_per_donation_cub <- (train_df$volume_per_donation)^3

# Let's create 4th degree power transformations of the predictor variables
train_df$Months.since.Last.Donation_four <- (train_df$Months.since.Last.Donation)^4
train_df$Number.of.Donations_four <- (train_df$Number.of.Donations)^4
train_df$Total.Volume.Donated..c.c.._four <- (train_df$Total.Volume.Donated..c.c..)^4
train_df$Months.since.First.Donation_four <- (train_df$Months.since.First.Donation)^4
train_df$num_months_per_donation_four <- (train_df$num_months_per_donation)^4
train_df$volume_per_donation_four <- (train_df$volume_per_donation)^4

# Let's create square root transformations of the predictor variables
train_df$Months.since.Last.Donation_sqrt <- sqrt(train_df$Months.since.Last.Donation)
train_df$Number.of.Donations_sqrt <- sqrt(train_df$Number.of.Donations)
train_df$Total.Volume.Donated..c.c.._sqrt <- sqrt(train_df$Total.Volume.Donated..c.c..)
train_df$Months.since.First.Donation_sqrt <- sqrt(train_df$Months.since.First.Donation)
train_df$num_months_per_donation_sqrt <- sqrt(train_df$num_months_per_donation)
train_df$volume_per_donation_sqrt <- sqrt(train_df$volume_per_donation)

# Let's create cubic root transformations of the predictor variables
train_df$Months.since.Last.Donation_cubroot <- (train_df$Months.since.Last.Donation)^(1/3)
train_df$Number.of.Donations_cubroot <- (train_df$Number.of.Donations)^(1/3)
train_df$Total.Volume.Donated..c.c.._cubroot <- (train_df$Total.Volume.Donated..c.c..)^(1/3)
train_df$Months.since.First.Donation_cubroot <- (train_df$Months.since.First.Donation)^(1/3)
train_df$num_months_per_donation_cubroot <- (train_df$num_months_per_donation)^(1/3)
train_df$volume_per_donation_cubroot <- (train_df$volume_per_donation)^(1/3)

# Let's create fourth root transformations of the predictor variables
train_df$Months.since.Last.Donation_fourthroot <- (train_df$Months.since.Last.Donation)^(1/4)
train_df$Number.of.Donations_fourthroot <- (train_df$Number.of.Donations)^(1/4)
train_df$Total.Volume.Donated..c.c.._fourthroot <- (train_df$Total.Volume.Donated..c.c..)^(1/4)
train_df$Months.since.First.Donation_fourthroot <- (train_df$Months.since.First.Donation)^(1/4)
train_df$num_months_per_donation_fourthroot <- (train_df$num_months_per_donation)^(1/4)
train_df$volume_per_donation_fourthroot <- (train_df$volume_per_donation)^(1/4)

# Let's discretize predictor variables into binary
train_df$Months.since.Last.Donation_binary <- (train_df$Months.since.Last.Donation) > mean(train_df$Months.since.Last.Donation)
train_df$Number.of.Donations_binary <- (train_df$Number.of.Donations) > mean(train_df$Number.of.Donations)
train_df$Total.Volume.Donated..c.c.._binary <- (train_df$Total.Volume.Donated..c.c..) > mean(train_df$Total.Volume.Donated..c.c..)
train_df$Months.since.First.Donation_binary <- (train_df$Months.since.First.Donation) > mean(train_df$Months.since.First.Donation)
train_df$num_months_per_donation_binary <- (train_df$num_months_per_donation) > mean(train_df$num_months_per_donation)
train_df$volume_per_donation_binary <- (train_df$volume_per_donation) > mean(train_df$volume_per_donation)

# Checking to see that last transformation worked!
summary(train_df$Months.since.Last.Donation_binary)

# Let's only apply Box Cox transformation to original vars
train_df_original_vars <- train_df[, 2:5]

# Now to Box cox Transformations

# trans <- preProcess(train_df_original_vars, method = c("BoxCox", "center", "scale", "pca"))

# Optional
# trans

# Apply the transformations:
# transformed <- predict(trans, train_df_original_vars)

# There are only 4 PCA variables here... We will add them to train_df
#train_df$PC1 <- transformed$PC1
#train_df$PC2 <- transformed$PC2
#train_df$PC3 <- transformed$PC3

# How about spatial sign transformation?
data_spatial <- data.frame(spatialSign(train_df_original_vars))

train_df$Months.since.Last.Donation_spatial <- data_spatial$Months.since.Last.Donation
train_df$Number.of.Donations_spatial <- data_spatial$Number.of.Donations
train_df$Total.Volume.Donated..c.c.._spatial <- data_spatial$Total.Volume.Donated..c.c..
train_df$Months.since.First.Donation_spatial <- data_spatial$Months.since.First.Donation

# I assume that the outcome as an integer with only 0,1 values is fine for binomial output

#Load these packages
library("arm")
library("caret")
library("arm")
library("caret")
library("class")
library("cvAUC")
library("e1071")
library("earth")
library("gam")
library("gbm")
library("genefilter")
library("ggplot2")
library("glmnet")
library("Hmisc")
library("ipred")
library("lattice")
library("LogicReg")
library("MASS")
library("mda")
library("mlbench")
library("nloptr")
library("nnet")
library("parallel")
library("party")
library("polspline")
library("quadprog")
library("randomForest")
library("ROCR")
library("rpart")
library("SIS")
library("spls")
library("stepPlr")
library("sva")
library('Rcpp')
library('arm')
library('gbm')
library('quantreg')

# Load the SuperLearner package with the library function.
library('SuperLearner')

# SL.library <- c("SL.glm", "SL.mean")

# SL.library <- c('SL.rpartPrune', 'SL.rpart', 'SL.logreg', 'SL.earth', 'SL.caret', 'SL.glm', 'SL.glm.interaction', 'SL.randomForest', 'SL.ipredbagg', 'SL.gam', 'SL.gbm', 'SL.nnet', 'SL.polymars', 'SL.bayesglm', 'SL.step', 'SL.step.interaction', 'SL.stepAIC','SL.leekasso', 'SL.svm', 'SL.glmnet', 'SL.knn', 'SL.mean')

SL.library <- list( c('SL.rpartPrune',  "screen.randomForest", "All", "screen.SIS"), c('SL.rpart',  "screen.randomForest", "All", "screen.SIS"), c('SL.logreg', "screen.randomForest", "All", "screen.SIS"), c('SL.earth',  "screen.randomForest", "All", "screen.SIS"), c('SL.caret',  "screen.randomForest", "All", "screen.SIS"), c('SL.glm.interaction',  "screen.randomForest", "All", "screen.SIS"), c('SL.ipredbagg',  "screen.randomForest", "All", "screen.SIS"), c('SL.gam',  "screen.randomForest", "All", "screen.SIS"), c('SL.gbm',  "screen.randomForest", "All", "screen.SIS"), c('SL.nnet',  "screen.randomForest", "All", "screen.SIS"), c('SL.bayesglm',  "screen.randomForest", "All", "screen.SIS"), c('SL.step',  "screen.randomForest", "All", "screen.SIS"), c('SL.step.interaction',  "screen.randomForest", "All", "screen.SIS"), c('SL.stepAIC',  "screen.randomForest", "All", "screen.SIS"), c('SL.leekasso',  "screen.randomForest", "All", "screen.SIS"), c('SL.svm',  "screen.randomForest", "All", "screen.SIS"), c('SL.knn', "screen.randomForest", "All", "screen.SIS"), c("SL.glmnet", "All"), c("SL.glm", "screen.randomForest", "All", "screen.SIS"), "SL.randomForest", c("SL.polymars", "All"), "SL.mean")

X <- subset(train_df, select=-c(X, Made.Donation.in.March.2007))

# first, let's initialize parallel package
library('parallel')

# Now, let's setup PSOCK multi-core cluster.
cl <- makeCluster(detectCores(), type = "PSOCK") # can use different types here
clusterSetRNGStream(cl, iseed = 2343)

# Let's do the analysis with snow SuperLearner. This should take advantage of multicore

# Note: the outcome Y must be a numeric vector
binary_outcome <- train_df$Made.Donation.in.March.2007

# Y = Donated (0/1)
# X = all predictors
# Family = binomial (binary outcomes)
# verbose = TRUE.  TRUE for printing progress during the computation (helpful for debugging).
# SL.library = machine learning libraries we use
# method = "NNloglik" for non-negative binomial likelihood maximization using the BFGS quasi-Newton optimization method
testSNOW <- snowSuperLearner(cluster = cl, Y = binary_outcome, X = X, SL.library = SL.library, family='binomial', verbose = TRUE, method = "method.CC_nloglik")

# Stop cluster once SuperLearner finishes running.
stopCluster(cl)

# Let's download the test set from CSV to a dataframe
test_df <- read.csv("Warm_Up_Predict_Blood_Donations_-_Test_Data.csv")

# NOw let's do feature extraction for test set...

# Simple Transformations

# Create new variable for how often a participant donates (in months)
test_df$num_months_per_donation <- (test_df$Months.since.First.Donation - test_df$Months.since.Last.Donation) / test_df$Number.of.Donations

# Create new variable to estimate how much blood volume is donated per number of donations
test_df$volume_per_donation <- (test_df$Total.Volume.Donated..c.c..)/(test_df$Number.of.Donations)

# Create new variable to estimate how much blood volume is donated per number of months
test_df$volume_per_donation <- (test_df$Total.Volume.Donated..c.c..)/(test_df$Months.since.First.Donation - test_df$Months.since.Last.Donation + 1)

# Let's create Log base 10 transformations of the predictor variables
test_df$Months.since.Last.Donation_log10 <- log10(test_df$Months.since.Last.Donation)
test_df$Number.of.Donations_log10 <- log10(test_df$Number.of.Donations)
test_df$Total.Volume.Donated..c.c.._log10 <- log10(test_df$Total.Volume.Donated..c.c..)
test_df$Months.since.First.Donation_log10 <- log10(test_df$Months.since.First.Donation)
test_df$num_months_per_donation_log10 <- log10(test_df$num_months_per_donation)
test_df$volume_per_donation_log10 <- log10(test_df$volume_per_donation)

# Let's create Log base 2 transformations of the predictor variables
test_df$Months.since.Last.Donation_log2 <- log2(test_df$Months.since.Last.Donation)
test_df$Number.of.Donations_log2 <- log2(test_df$Number.of.Donations)
test_df$Total.Volume.Donated..c.c.._log2 <- log2(test_df$Total.Volume.Donated..c.c..)
test_df$Months.since.First.Donation_log2 <- log2(test_df$Months.since.First.Donation)
test_df$num_months_per_donation_log2 <- log2(test_df$num_months_per_donation)
test_df$volume_per_donation_log2 <- log2(test_df$volume_per_donation)

# Let's create exponential transformations of the predictor variables
test_df$Months.since.Last.Donation_exp <- exp(test_df$Months.since.Last.Donation)
test_df$Number.of.Donations_exp <- exp(test_df$Number.of.Donations)
test_df$Total.Volume.Donated..c.c.._exp <- exp(test_df$Total.Volume.Donated..c.c..)
test_df$Months.since.First.Donation_exp <- exp(test_df$Months.since.First.Donation)
test_df$num_months_per_donation_exp <- exp(test_df$num_months_per_donation)
test_df$volume_per_donation_exp <- exp(test_df$volume_per_donation)

# Let's create sine transformations of the predictor variables
test_df$Months.since.Last.Donation_sin <- sin(test_df$Months.since.Last.Donation)
test_df$Number.of.Donations_sin <- sin(test_df$Number.of.Donations)
test_df$Total.Volume.Donated..c.c.._sin <- sin(test_df$Total.Volume.Donated..c.c..)
test_df$Months.since.First.Donation_sin <- sin(test_df$Months.since.First.Donation)
test_df$num_months_per_donation_sin <- sin(test_df$num_months_per_donation)
test_df$volume_per_donation_sin <- sin(test_df$volume_per_donation)

# Let's create cosine transformations of the predictor variables
test_df$Months.since.Last.Donation_cos <- cos(test_df$Months.since.Last.Donation)
test_df$Number.of.Donations_cos <- cos(test_df$Number.of.Donations)
test_df$Total.Volume.Donated..c.c.._cos <- cos(test_df$Total.Volume.Donated..c.c..)
test_df$Months.since.First.Donation_cos <- cos(test_df$Months.since.First.Donation)
test_df$num_months_per_donation_cos <- cos(test_df$num_months_per_donation)
test_df$volume_per_donation_cos <- cos(test_df$volume_per_donation)

# Let's create tangent transformations of the predictor variables
test_df$Months.since.Last.Donation_tan <- tan(test_df$Months.since.Last.Donation)
test_df$Number.of.Donations_tan <- tan(test_df$Number.of.Donations)
test_df$Total.Volume.Donated..c.c.._tan <- tan(test_df$Total.Volume.Donated..c.c..)
test_df$Months.since.First.Donation_tan <- tan(test_df$Months.since.First.Donation)
test_df$num_months_per_donation_tan <- tan(test_df$num_months_per_donation)
test_df$volume_per_donation_tan <- tan(test_df$volume_per_donation)

# Let's create Squared transformations of the predictor variables
test_df$Months.since.Last.Donation_sq <- (test_df$Months.since.Last.Donation)^2
test_df$Number.of.Donations_sq <- (test_df$Number.of.Donations)^2
test_df$Total.Volume.Donated..c.c.._sq <- (test_df$Total.Volume.Donated..c.c..)^2
test_df$Months.since.First.Donation_sq <- (test_df$Months.since.First.Donation)^2
test_df$num_months_per_donation_sq <- (test_df$num_months_per_donation)^2
test_df$volume_per_donation_sq <- (test_df$volume_per_donation)^2

# Let's create Cubic transformations of the predictor variables
test_df$Months.since.Last.Donation_cub <- (test_df$Months.since.Last.Donation)^3
test_df$Number.of.Donations_cub <- (test_df$Number.of.Donations)^3
test_df$Total.Volume.Donated..c.c.._cub <- (test_df$Total.Volume.Donated..c.c..)^3
test_df$Months.since.First.Donation_cub <- (test_df$Months.since.First.Donation)^3
test_df$num_months_per_donation_cub <- (test_df$num_months_per_donation)^3
test_df$volume_per_donation_cub <- (test_df$volume_per_donation)^3

# Let's create 4th degree power transformations of the predictor variables
test_df$Months.since.Last.Donation_four <- (test_df$Months.since.Last.Donation)^4
test_df$Number.of.Donations_four <- (test_df$Number.of.Donations)^4
test_df$Total.Volume.Donated..c.c.._four <- (test_df$Total.Volume.Donated..c.c..)^4
test_df$Months.since.First.Donation_four <- (test_df$Months.since.First.Donation)^4
test_df$num_months_per_donation_four <- (test_df$num_months_per_donation)^4
test_df$volume_per_donation_four <- (test_df$volume_per_donation)^4

# Let's create square root transformations of the predictor variables
test_df$Months.since.Last.Donation_sqrt <- sqrt(test_df$Months.since.Last.Donation)
test_df$Number.of.Donations_sqrt <- sqrt(test_df$Number.of.Donations)
test_df$Total.Volume.Donated..c.c.._sqrt <- sqrt(test_df$Total.Volume.Donated..c.c..)
test_df$Months.since.First.Donation_sqrt <- sqrt(test_df$Months.since.First.Donation)
test_df$num_months_per_donation_sqrt <- sqrt(test_df$num_months_per_donation)
test_df$volume_per_donation_sqrt <- sqrt(test_df$volume_per_donation)

# Let's create cubic root transformations of the predictor variables
test_df$Months.since.Last.Donation_cubroot <- (test_df$Months.since.Last.Donation)^(1/3)
test_df$Number.of.Donations_cubroot <- (test_df$Number.of.Donations)^(1/3)
test_df$Total.Volume.Donated..c.c.._cubroot <- (test_df$Total.Volume.Donated..c.c..)^(1/3)
test_df$Months.since.First.Donation_cubroot <- (test_df$Months.since.First.Donation)^(1/3)
test_df$num_months_per_donation_cubroot <- (test_df$num_months_per_donation)^(1/3)
test_df$volume_per_donation_cubroot <- (test_df$volume_per_donation)^(1/3)

# Let's create fourth root transformations of the predictor variables
test_df$Months.since.Last.Donation_fourthroot <- (test_df$Months.since.Last.Donation)^(1/4)
test_df$Number.of.Donations_fourthroot <- (test_df$Number.of.Donations)^(1/4)
test_df$Total.Volume.Donated..c.c.._fourthroot <- (test_df$Total.Volume.Donated..c.c..)^(1/4)
test_df$Months.since.First.Donation_fourthroot <- (test_df$Months.since.First.Donation)^(1/4)
test_df$num_months_per_donation_fourthroot <- (test_df$num_months_per_donation)^(1/4)
test_df$volume_per_donation_fourthroot <- (test_df$volume_per_donation)^(1/4)

# Let's discretize predictor variables into binary
test_df$Months.since.Last.Donation_binary <- (test_df$Months.since.Last.Donation) > mean(test_df$Months.since.Last.Donation)
test_df$Number.of.Donations_binary <- (test_df$Number.of.Donations) > mean(test_df$Number.of.Donations)
test_df$Total.Volume.Donated..c.c.._binary <- (test_df$Total.Volume.Donated..c.c..) > mean(test_df$Total.Volume.Donated..c.c..)
test_df$Months.since.First.Donation_binary <- (test_df$Months.since.First.Donation) > mean(test_df$Months.since.First.Donation)
test_df$num_months_per_donation_binary <- (test_df$num_months_per_donation) > mean(test_df$num_months_per_donation)
test_df$volume_per_donation_binary <- (test_df$volume_per_donation) > mean(test_df$volume_per_donation)

# Checking to see that last transformation worked!
summary(test_df$Months.since.Last.Donation_binary)

# Let's only apply Box Cox transformation to original vars
test_df_original_vars <- test_df[, 2:5]

# Now to Box cox Transformations

#trans_test <- preProcess(test_df_original_vars, method = c("BoxCox", "center", "scale", "pca"))

# Optional
# trans

# Apply the transformations:
#transformed_test <- predict(trans_test, test_df_original_vars)

# There are only 4 PCA variables here... We will add them to test_df
#test_df$PC1 <- transformed_test$PC1
#test_df$PC2 <- transformed_test$PC2
#test_df$PC3 <- 0

# How about spatial sign transformation?
data_spatial_test <- data.frame(spatialSign(test_df_original_vars))

test_df$Months.since.Last.Donation_spatial <- data_spatial_test$Months.since.Last.Donation
test_df$Number.of.Donations_spatial <- data_spatial_test$Number.of.Donations
test_df$Total.Volume.Donated..c.c.._spatial <- data_spatial_test$Total.Volume.Donated..c.c..
test_df$Months.since.First.Donation_spatial <- data_spatial_test$Months.since.First.Donation

# Only extract variables from test set for prediction and place them in dataframe newdata 
newdata <- subset(test_df, select=-X)

# let's predict blood donation based on SuperLearner object on newdata
blood_donation_prediction <- predict(object=testSNOW, newdata=newdata, onlySL=TRUE)

test_df$Made.Donation.in.March.2007 <- blood_donation_prediction$pred

header_row <- c("", "Made Donation in March 2007")

# Let's turn this into a CSV file for submission
submit1 <- data.frame(test_df$X, test_df$Made.Donation.in.March.2007)

submit1 <- rbind(header_row, submit1)

# Write to CSV. Use write.table to ignore row/col names
write.table(submit1, file = "submit_transform_CC_nloglik.csv", sep=",", row.names=FALSE, col.names=FALSE)