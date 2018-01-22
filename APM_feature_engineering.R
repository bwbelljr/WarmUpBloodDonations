# APM Chapter 3

update.packages("AppliedPredictiveModeling")

library("AppliedPredictiveModeling")

# The function apropos will search any loaded R packages for a given term. For example, to find functions for creating a confusion matrix within the currently 
# loaded packages:

apropos("confusion")

# To find such a function in any package, the RSiteSearch function can help. Running the command:

RSiteSearch("confusion", restrict = "functions")

# will search online to find matches and will open a web browser to display the results.

# The raw segmentation data set is contained in the AppliedPredictiveModeling package. To load the data set into R:
data(segmentationOriginal) # I did it manually!

# There were fields that identified each cell (called Cell) and a factor vector
# that indicated which cells were well segmented (Class). The variable Case
# indicated which cells were originally used for the training and test sets. The
# analysis in this chapter focused on the training set samples, so the data are
# filtered for these cells:

segData <- subset(segmentationOriginal, Case == "Train")

# The Class and Cell fields will be saved into separate vectors, then removed from the main object:

cellID <- segData$Cell
class <- segData$Class
case <- segData$Case
# Now remove the columns
segData <- segData[, -(1:3)]

# The original data contained several “status” columns which were binary versions
# of the predictors. To remove these, we find the column names containing
# "Status" and remove them:

statusColNum <- grep("Status", names(segData))
statusColNum
segData <- segData[, -statusColNum]


# Transformations
# As previously discussed, some features exhibited significantly skewness. The
# skewness function in the e1071 package calculates the sample skewness statistic
# for each predictor:

library(e1071)
#For one predictor:
skewness(segData$AngleCh1)

# Since all the predictors are numeric columns, the apply function can
# be used to compute the skewness across columns.
skewValues <- apply(segData, 2, skewness)
head(skewValues)

# Using these values as a guide, the variables can be prioritized for visualizing
# the distribution. The basic R function hist or the histogram function in the
# lattice can be used to assess the shape of the distribution.
# To determine which type of transformation should be used, the MASS
# package contains the boxcox function. Although this function estimates λ, it
# does not create the transformed variable(s). A caret function, BoxCoxTrans,
# can find the appropriate transformation and apply them to the new data:

library(caret)
Ch1AreaTrans <- BoxCoxTrans(segData$AreaCh1)
Ch1AreaTrans

# The original data
head(segData$AreaCh1)

# After transformation
predict(Ch1AreaTrans, head(segData$AreaCh1))

# To turn the transformation into a vector, do the following:
AreaCh1_BoxCox <- predict(Ch1AreaTrans, segData$AreaCh1)
head(AreaCh1_BoxCox)

# This is the actual calculation, using first value and lambda
(819^(-.9) - 1)/(-.9)

# Another caret function, preProcess, applies this transformation to a set of
# predictors. This function is discussed below. The base R function prcomp can
# be used for PCA. In the code below, the data are centered and scaled prior to PCA.

?prcomp

pcaObject <- prcomp(segData, center = TRUE, scale. = TRUE)
# Calculate the cumulative percentage of variance which each component
# accounts for.
percentVariance <- pcaObject$sd^2/sum(pcaObject$sd^2)*100
percentVariance[1:3]

# The transformed values are stored in pcaObject as a sub-object called x:
head(pcaObject$x[, 1:5])

summary(pcaObject$x[,10])


# The another sub-object called rotation stores the variable loadings, where
# rows correspond to predictor variables and columns are associated with the
# components:
head(pcaObject$rotation[, 1:3])

# These two transformations are different
summary(pcaObject$rotation[, 1])
summary(pcaObject$x[, 1])

# The caret package class spatialSign contains functionality for the spatial sign
# transformation. Although we will not apply this technique to these data, the
# basic syntax would be spatialSign(segData).

segData_spatial <- spatialSign(segData)

# To administer a series of transformations to multiple data sets, the caret
# class preProcess has the ability to transform, center, scale, or impute values,
# as well as apply the spatial sign transformation and feature extraction. The
# function calculates the required quantities for the transformation. After calling
# the preProcess function, the predict method applies the results to a set
# of data. For example, to Box–Cox transform, center, and scale the data, then
# execute PCA for signal extraction, the syntax would be:

trans <- preProcess(segData, method = c("BoxCox", "center", "scale", "pca"))

trans

# Apply the transformations:
transformed <- predict(trans, segData)
# These values are different than the previous PCA components since
# they were transformed prior to PCA
head(transformed[, 1:5])




