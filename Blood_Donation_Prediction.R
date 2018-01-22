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

SL.library <- list('SL.rpartPrune', 'SL.rpart', 'SL.logreg', 'SL.earth', 'SL.caret', 'SL.glm.interaction', 'SL.ipredbagg', 'SL.gam', 'SL.gbm', 'SL.nnet', 'SL.bayesglm', 'SL.step', 'SL.step.interaction', 'SL.stepAIC', 'SL.leekasso', 'SL.svm', 'SL.knn', c("SL.glmnet", "All"), c("SL.glm", "screen.randomForest", "All", "screen.SIS"), "SL.randomForest", c("SL.polymars", "All"), "SL.mean")

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
testSNOW <- snowSuperLearner(cluster = cl, Y = binary_outcome, X = X, SL.library = SL.library, family='binomial', verbose = TRUE, method = "method.NNloglik")

# Stop cluster once SuperLearner finishes running.
stopCluster(cl)

# Let's download the test set from CSV to a dataframe
test_df <- read.csv("Warm_Up_Predict_Blood_Donations_-_Test_Data.csv")

# Only extract variables from test set for prediction and place them in dataframe newdata 
newdata <- subset(test_df, select=-X)

# let's predict blood donation based on SuperLearner object on newdata
blood_donation_prediction <- predict(object=testSNOW, newdata=newdata)

test_df$Made.Donation.in.March.2007 <- blood_donation_prediction$pred

header_row <- c("", "Made Donation in March 2007")

# Let's turn this into a CSV file for submission
submit1 <- data.frame(test_df$X, test_df$Made.Donation.in.March.2007)

submit1 <- rbind(header_row, submit1)

# Write to CSV. Use write.table to ignore row/col names
write.table(submit1, file = "submit1.csv", sep=",", row.names=FALSE, col.names=FALSE)