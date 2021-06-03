# Load required libraries

library("leaps")
library("ggplot2")
library("reshape2")
library("MASS")
library("ggcorrplot")
library("plotmo")
library("dplyr")
library("gridExtra")
library("Simpsons")
library("GGally")
library("memisc")
library("pander")
library("caret")
library("glmnet")
library("mlbench")
library("psych")


# Load and view the dataset
whitewine = read.csv("https://raw.githubusercontent.com/vicky61992/StatisticProject_winequality/main/winequality-white.csv", sep = ";", header = T)
redwine = read.csv("https://raw.githubusercontent.com/vicky61992/StatisticProject_winequality/main/winequality-red.csv", sep = ";", header = T)

View(whitewine)

View(redwine)

head(whitewine)

head(redwine)




# sortcut function to sort the data frame column


sortByCorr = function(dataset, refColName) {
  refColIdx = grep(refColName, colnames(dataset))
  corrTmp = cor(dataset)[, refColIdx]
  corrTmp[order(abs(corrTmp), decreasing = TRUE)]
  
  dataset[, order(abs(corrTmp), decreasing = TRUE)]
}

# Discriptive analysis

dim(whitewine) # the whitewine dataset has 4898 obs., 11 predictors and 1 output

dim(redwine) # the redwine dataset has 1599 obs., 11 predictors and 1 output

summary(whitewine)

summary(redwine)

sapply(whitewine,function(x)sum(is.na(x)))

sapply(redwine,function(x)sum(is.na(x)))

sapply(whitewine,class)

sapply(redwine, class)

str(whitewine)

str(redwine)


# Boxplot for each variables 

oldpar = par(mfrow = c(2,6))
for ( i in 1:11 ) {
  boxplot(whitewine[[i]])
  mtext(names(whitewine)[i], cex = 0.8, side = 1, line = 2)
}
par(oldpar)

# Boxplot shows that all the variable except alcohol have outliers in whitewine dataset.

oldpar1 = par(mfrow = c(2,6))
for ( i in 1:11 ) {
  boxplot(redwine[[i]])
  mtext(names(redwine)[i], cex = 0.8, side = 1, line = 2)
}
par(oldpar1)

# Boxplot shows that all variable contains outliers, citric.acid contains very less.

# Scatter plot matrix for more insights

pairs(whitewine[, -grep("quality", colnames(whitewine))])

pairs(redwine[, -grep("quality", colnames(redwine))])


# Predictor values Histogram distribution

oldpar = par(mfrow = c(4,3))
for ( i in 1:12 ) {
  truehist(whitewine[[i]], xlab = names(whitewine)[i],
           col = 'red', main = paste("Average =", signif(mean(whitewine[[i]]),3)))
}

# We see in dataset all variables has left skewed except pH & quality (both are normally distributed)

oldpar1 = par(mfrow = c(4,3))
for ( i in 1:12 ) {
  truehist(redwine[[i]], xlab = names(redwine)[i],
           col = 'blue', main = paste("Average =", signif(mean(redwine[[i]]),5)))
}

# We see in dataset all variables has left skewed except quality (normally distributed)




## Outlier Detection 


outliers = c()
for ( i in 1:11 ) {
  stats = boxplot.stats(whitewine[[i]])$stats
  bottom_outlier_rows = which(whitewine[[i]] < stats[1])
  top_outlier_rows = which(whitewine[[i]] > stats[5])
  outliers = c(outliers , top_outlier_rows[ !top_outlier_rows %in% outliers ] )
  outliers = c(outliers , bottom_outlier_rows[ !bottom_outlier_rows %in% outliers ] )
}


# Cook's distance to detect influential observations.

mod = lm(quality ~ ., data = whitewine)
cooksd = cooks.distance(mod)
plot(cooksd, pch = "*", cex = 2, main = "Influential Obs by Cooks distance")
abline(h = 4*mean(cooksd, na.rm = T), col = "red")


head(whitewine[cooksd > 4 * mean(cooksd, na.rm=T), ])


# looking at each row we can find out why it is influential
# Row 99 & 252 have very high residual.sugar
# Row 251 & 252 have high free.sulfur.dioxide
# Row 251 & 252 have high sulfur.dioxide
# Row 99 & 254 have low free.sulfur.dioxide


# We remove all the ouliers in our list from the dataset and create a new set of histograms:

coutliers = as.numeric(rownames(whitewine[cooksd > 4 * mean(cooksd, na.rm=T), ]))
outliers = c(outliers , coutliers[ !coutliers %in% outliers ] )

cleanWhitewine = whitewine[-outliers, ]
oldpar = par(mfrow=c(4,3))
for ( i in 1:12 ) {
  truehist(cleanWhitewine[[i]], xlab = names(cleanWhitewine)[i],
           col = 'green', main = paste("Average =", signif(mean(cleanWhitewine[[i]]),7)))
}

par(oldpar)

# After removing all the outliers, all the variables are normally distributed except residual.sugar
# After removing of all outliers our dataset size is 3999 obs. and 12 variables


# Scatter plot matrice  on clean dataset to determined the colleration bet all varibales.

heatmap(as.matrix(cleanWhitewine), scale = "column",
        col= heat.colors(256),
        main = "HeatMap",
        Rowv = NA,
        Colv = NA)


pairs(cleanWhitewine, col = cleanWhitewine$quality, pch = cleanWhitewine$quality)

pairs(cleanWhitewine[,c(7, 8, 10, 11)], col = cleanWhitewine$quality, pch = cleanWhitewine$quality)




## Outliear detection for red wine 

outliers = c()
for ( i in 1:11 ) {
  stats = boxplot.stats(redwine[[i]])$stats
  bottom_outlier_rows = which(redwine[[i]] < stats[1])
  top_outlier_rows = which(redwine[[i]] > stats[5])
  outliers = c(outliers , top_outlier_rows[ !top_outlier_rows %in% outliers ] )
  outliers = c(outliers , bottom_outlier_rows[ !bottom_outlier_rows %in% outliers ] )
}


# Cook's distance to detect influential observations.

mod = lm(quality ~ ., data = redwine)
cooksd = cooks.distance(mod)
plot(cooksd, pch = "*", cex = 2, main = "Influential Obs by Cooks distance")
abline(h = 4*mean(cooksd, na.rm = T), col = "blue")


head(redwine[cooksd > 4 * mean(cooksd, na.rm=T), ])


# looking at each row we can find out why it is influential
# Row 34 have very high residual.sugar
# Row 34 have high free.sulfur.dioxide
# Row 80,87& 92 have high sulfur.dioxide
# Row 14 & 8 have low free.sulfur.dioxide


# We remove all the ouliers in our list from the dataset and create a new set of histograms:

coutliers1 = as.numeric(rownames(redwine[cooksd > 4 * mean(cooksd, na.rm=T), ]))
outliers1 = c(outliers , coutliers[ !coutliers %in% outliers ] )

cleanRedwine = redwine[-outliers1, ]
oldpar1 = par(mfrow=c(4,3))
for ( i in 1:12 ) {
  truehist(cleanRedwine[[i]], xlab = names(cleanRedwine)[i],
           col = 'yellow', main = paste("Average =", signif(mean(cleanRedwine[[i]]),7)))
}

par(oldpar1)

# After removing all the outliers, dataset size is 1174 obs. and 12 variables
# After removing of outliers some variables are normally distributed or some are not.


# Scatter plot matrice  on clean dataset to determined the colleration bet all varibales.
heatmap(as.matrix(cleanRedwine), scale = "column",
        col= heat.colors(256),
        main = "HeatMap",
        Rowv = NA,
        Colv = NA)



pairs(cleanRedwine, col = cleanRedwine$quality, pch = cleanRedwine$quality)

pairs(cleanRedwine[,c(7, 8, 10, 11)], col = cleanRedwine$quality, pch = cleanRedwine$quality)

## Correlation for whitewine

ggcorrplot(cor(cleanWhitewine), hc.order = TRUE, type = "lower", lab = TRUE, insig = "blank")

# there are strong correlation between density/residual.sugar & density/alcohol

colnames(sortByCorr(dataset = cleanWhitewine, refColName = 'quality'))


## Correlation for redwine

ggcorrplot(cor(cleanRedwine), hc.order = TRUE, type = "lower", lab = TRUE, insig = "blank")

# there are strong positive correlation between density/fixed.acidity,quality/alcohol,critric.acid/fixed.acidity & free.sulfur.dioxide/ total.sulfur.dioxide
# there are strong negative correlation between pH/fixed.acidity, volatile.acidity/critric.acid & alcohol/density

colnames(sortByCorr(dataset = cleanRedwine, refColName = 'quality'))

# Model

whiteFit = lm(quality~., cleanWhitewine)

summary(whiteFit)


redFit = lm(quality~., cleanRedwine)

summary(redFit)

# Calculate the MSE

mean(whiteFit$residuals^2)

mean(redFit$residuals^2)


#Linear Regression 

# Data Partition
set.seed(222)
ind <- floor(0.75*nrow(cleanWhitewine))
print(ind)

train_ind <- sample(seq_len(nrow(cleanWhitewine)),size = ind)

print(train_ind)

test = -train_ind
test

train_data<-cleanWhitewine[train_ind,]
test_data<-cleanWhitewine[-train_ind,]

model1<- lm(quality~.,data= train_data)

summary(model1)

# Linear Prediction

predtest<- predict(model1,test_data)

predtest1<-data.frame(predtest)
final_data<- cbind(test_data,predtest1)
write.csv(final_data,"winequality.csv")


# Ridge Regression
# Data Partition
set.seed(222)
ind <- sample(2, nrow(cleanWhitewine), replace = T, prob = c(0.7, 0.3))
train <- cleanWhitewine[ind==1,]
test <- cleanWhitewine[ind==2,]

# Custom Control Parameters
custom <- trainControl(method = "repeatedcv",
                       number = 10,
                       repeats = 5,
                       verboseIter = T)


#Ridge Regresion

set.seed(1234)
ridge <- train(quality~.,
               data= train,
               method = "glmnet",
               tuneGrid = expand.grid(alpha = 0,
                                      lambda = seq(0.0001,1, length = 5)),
               trControl = custom)
plot(ridge)
plot(ridge$finalModel, xvar = "lambda", label = T)
plot(ridge$finalModel, xvar = "dev" , label = T)
plot(varImp(ridge, scale = T))

#Predict


p1<- predict(ridge,train)
sqrt(mean((train$quality-p1)^2))


p2<- predict(ridge,test)
sqrt(mean((test$quality-p2)^2))



predtest2<-data.frame(p2)
final_data1<- cbind(test,predtest2)
write.csv(final_data,"winequality_ridge.csv")









































































