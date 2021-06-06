setwd("C:/Users/VSBAG/Desktop/DSE_Milan/3rd_sem_subject/Stat/Group_Project")
rm(list = ls())
# For aesthetic purposes, I usually put this part in a separate .R file, which can be called through the source() function. I am displaying it here just for the purpose of showing all used code.

packages = c("tidyverse", "RCurl", "psych", "stats", 
             "randomForest", "glmnet", "caret","kernlab", 
             "rpart", "rpart.plot", "neuralnet", "C50",
             "doParallel", "AUC", "ggfortify")
if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
  install.packages(setdiff(packages, rownames(installed.packages())))  
}
invisible(lapply(packages, require, character.only = TRUE))

# customized function to evaluate model performance for continuous predictors
eval = function(pred, true, plot = F, title = "") {
  rmse = sqrt(mean((pred - true)^2))
  mae = mean(abs(pred - true))
  cor = cor(pred, true)
  if (plot == TRUE) {
    par(mfrow = c(1,2), oma = c(0, 0, 2, 0))
    diff = pred - true
    plot(jitter(true, factor = 1), 
         jitter(pred, factor = 0.5), #jitter so that we can see overlapped dots
         pch = 3, asp = 1,
         xlab = "Truth", ylab = "Predicted") 
    abline(0,1, lty = 2)
    hist(diff, breaks = 20, main = NULL)
    mtext(paste0(title, " predicted vs. true using test set"), outer = TRUE)
    par(mfrow = c(1,1))}
  return(list(rmse = rmse,
              mae = mae,
              cor = cor))
}
# customized function to evaluate model performance for binary predictors
eval_class = function(prob, true, plot = F, title = "") {
  # find cutoff with the best kappa
  cuts = seq(0.01, 0.99, by=0.01)
  kappa = c()
  for (cut in cuts){
    cat = as.factor(ifelse(prob >= cut, 1, 0))
    cm = confusionMatrix(cat, true, positive = "1")
    kappa = c(kappa, cm$overall[["Kappa"]])
  }
  opt.cut = cuts[which.max(kappa)]
  
  # make predictions based on best kappa
  pred = as.factor(ifelse(prob >= opt.cut, 1, 0))
  confM = confusionMatrix(pred, true, positive = "1")
  
  # calculate AUC
  roc = roc(as.vector(prob), as.factor(true))
  auc = round(AUC::auc(roc),3)
  
  if (plot==T){
    # plot area under the curve
    par(mfrow = c(1,2), oma = c(0, 0, 2, 0))
    plot(roc, main = "AUC curve"); abline(0,1)
    text(0.8, 0.2, paste0("AUC = ", auc))
    
    # plot confusion matrix
    tab = table(true, pred)
    plot(tab,
         xlab = "Truth",
         ylab = "Predicted",
         main = "Confusion Matrix")
    text(0.9, 0.9, paste0('FN:', tab[2,1]))
    text(0.9, 0.05, paste0('TP:', tab[2,2]))
    text(0.1, 0.9, paste0('TN:', tab[1,1]))
    text(0.1, 0.05, paste0('FP:', tab[1,2]))
    mtext(paste0(title, " predicted vs. true using test set"), outer = TRUE)
    par(mfrow = c(1,1))
  }
  return(list(auc=auc, 
              confusionMatrix = confM))
}

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

# Data importing and processing

redwine = read.csv("https://raw.githubusercontent.com/vicky61992/StatisticProject_winequality/main/winequality-red.csv", sep = ";", header = T)

dim(redwine)
str(redwine)
summary(redwine)
sapply(redwine,function(x)sum(is.na(x)))
pairs.panels(redwine)




# sortcut function to sort the data frame column


sortByCorr = function(dataset, refColName) {
  refColIdx = grep(refColName, colnames(dataset))
  corrTmp = cor(dataset)[, refColIdx]
  corrTmp[order(abs(corrTmp), decreasing = TRUE)]
  
  dataset[, order(abs(corrTmp), decreasing = TRUE)]
}

# Boxplot for each variables in redwine

oldpar = par(mfrow = c(2,6))
for ( i in 1:11 ) {
  boxplot(redwine[[i]])
  mtext(names(redwine)[i], cex = 0.8, side = 1, line = 2)
}
par(oldpar)

# Boxplot shows that all variable contains outliers, citric.acid contains very less.

# Scatter plot matrix for more insights

pairs(redwine[, -grep("quality", colnames(redwine))])

# Predictor values Histogram distribution of redwine

oldpar = par(mfrow = c(4,3))
for ( i in 1:12 ) {
  truehist(redwine[[i]], xlab = names(redwine)[i],
           col = 'blue', main = paste("Average =", signif(mean(redwine[[i]]),5)))
}

# We see in dataset all variables has left skewed except quality (normally distributed)

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

coutliers = as.numeric(rownames(redwine[cooksd > 4 * mean(cooksd, na.rm=T), ]))
outliers = c(outliers , coutliers[ !coutliers %in% outliers ] )

cleanRedwine = redwine[-outliers, ]
oldpar = par(mfrow=c(4,3))
for ( i in 1:12 ) {
  truehist(cleanRedwine[[i]], xlab = names(cleanRedwine)[i],
           col = 'yellow', main = paste("Average =", signif(mean(cleanRedwine[[i]]),7)))
}

par(oldpar)

dim(cleanRedwine)


# After removing all the outliers, dataset size is 1179 obs. and 12 variables
# After removing of outliers some variables are normally distributed or some are not.

# Boxplot for each variables in cleanRedwine

oldpar = par(mfrow = c(2,6))
for ( i in 1:11 ) {
  boxplot(cleanRedwine[[i]])
  mtext(names(cleanRedwine)[i], cex = 0.8, side = 1, line = 2)
}
par(oldpar)



# Scatter plot matrice  on clean dataset to determined the colleration bet all varibales.

pairs(cleanRedwine, col = cleanRedwine$quality, pch = cleanRedwine$quality)


# HeatMap
res<- cor(cleanRedwine)
col1<- colorRampPalette(c("blue", "white", "red"))(20)
heatmap(x = res, col = col1, symm = TRUE, Colv = NA, Rowv = NA)

## Correlation for redwine

ggcorrplot(cor(cleanRedwine), hc.order = TRUE, type = "lower", lab = TRUE, insig = "blank")

# there are strong positive correlation between density/fixed.acidity,quality/alcohol,critric.acid/fixed.acidity & free.sulfur.dioxide/ total.sulfur.dioxide
# there are strong negative correlation between pH/fixed.acidity, volatile.acidity/critric.acid & alcohol/density

colnames(sortByCorr(dataset = cleanRedwine, refColName = 'quality'))

# Model for Redwine

redFit = lm(quality~., cleanRedwine)

summary(redFit)

# As per model summary volatile.acidity , total.sulfur.dioxide,sulphates & alcohal are significant


# Data Partition
set.seed(222)
ind <- floor(0.75*nrow(cleanRedwine))
print(ind)

train_ind <- sample(seq_len(nrow(cleanRedwine)),size = ind)

print(train_ind)

test = -train_ind
test

train<-cleanRedwine[train_ind,]
test<-cleanRedwine[-train_ind,]

dim(train)
dim(test)


tr.lm = lm(quality~., data = train)
summary(tr.lm)

tr.lm.pred = predict(tr.lm, test[,-12])
tr.lm.eval = eval(tr.lm.pred, test$quality, plot = T, title = "lm: "); unlist(tr.lm.eval)


scatter.smooth(x = test$quality ,tr.lm.pred )



# normalize train set so that the range is 0 ~ 1

normalize_train = function(x){
  return ((x-min(x, na.rm = TRUE))/(max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))}

train.norm = data.frame(apply(train[1:11],2,normalize_train),
                        quality = train[12] )
                          
summary(train.norm)


# normalize test set using the values from train set to make prediction comparable
train.min = apply(train[,-12], 2, min)
train.max = apply(train[,-12], 2, max)
test.norm = data.frame(sweep(test, 2, c(train.min, 0)) %>% 
                         sweep(2, c(train.max-train.min, 1), FUN = "/"))
summary(test.norm) # test.norm might have data out of range 0~1, since it's normalized against the training set.

#Linear Regression on Redwine

hist(cleanRedwine$quality , main = "histogram of Wine Quality")

shapiro.test(cleanRedwine$quality) #Didn't pass normality test, so linear model may have a problem

#The dependent variable doesn't pass the normality test, so one assumption of linear regression is not met. In addition, as we see from the pairwise plot, the relationship among independent variables and dependent variables are not entirely linear. There is also some collinearity among independent variables. Any of those could sabotage the performance of the linear model.

#Then I will apply this linear model to the test set, and visualize the predicted value against the true value. I will also evaluate the model performance based on 3 measures: RMSE (root mean square error), MAE (mean absolute error) and cor (correlation). Smaller RMSE, MAE and larger cor are indicators of a good prediction.

# Linear Regression on normalize data
tr.lm = lm(quality~., data = train.norm)
summary(tr.lm)

tr.lm.pred = predict(tr.lm, test.norm[,-12])
tr.lm.eval = eval(tr.lm.pred, test.norm$quality, plot = T, title = "lm: "); unlist(tr.lm.eval)


scatter.smooth(x = test.norm$quality ,tr.lm.pred )


# Outlier/influential point detection

par(mfrow=c(2,3))
lapply(1:6, function(x) plot(tr.lm, which=x, labels.id= 1:nrow(train.norm))) %>% invisible()


rm = c(824,867)
removed = train.norm[rm, ];removed # these observations will be removed from the training set

train.norm = train.norm[-rm, ]


# Linear Regression on normalize data after removing outlier
tr.lm = lm(quality~., data = train.norm)
summary(tr.lm)

tr.lm.pred = predict(tr.lm, test.norm[,-12])
tr.lm.eval = eval(tr.lm.pred, test.norm$quality, plot = T, title = "lm: "); unlist(tr.lm.eval)


scatter.smooth(x = test.norm$quality ,tr.lm.pred )



# Polynomial Regression

# 2nd order regression (quadratic model)
tr.qm = lm(quality~ poly(fixed.acidity, 2) + 
             poly(volatile.acidity,2) + 
             poly(citric.acid,2) + 
             poly(residual.sugar,2) +  
             poly(chlorides,2) + 
             poly(free.sulfur.dioxide,2) +
             poly(total.sulfur.dioxide,2) + 
             poly(density,2) + 
             poly(pH,2) + 
             poly(sulphates,2) + 
             poly(alcohol,2), 
           data = train.norm)
summary(tr.qm)

tr.qm.pred = predict(tr.qm, test.norm[,-12])
tr.qm.eval = eval(tr.qm.pred, test.norm$quality, plot=T, title="quadratic model: ");unlist(tr.qm.eval)

scatter.smooth(x = test.norm$quality ,tr.qm.pred )



# Variable interaction and variable selection

tr.lm.interract = lm(quality~ .^2, data = train.norm)
summary(tr.lm.interract)

# variable selection using stepwise methods
lm0 = lm(quality ~ 1, data = train.norm)
tr.lm.interract.step = step(lm0, ~ (fixed.acidity + volatile.acidity + 
                                      citric.acid + residual.sugar +  chlorides + free.sulfur.dioxide +
                                      total.sulfur.dioxide + density + pH + sulphates + alcohol)^2, 
                            direction = "both", trace = 0)
summary(tr.lm.interract.step)


tr.lm.interract.step.pred = predict(tr.lm.interract.step, test.norm[,-12])
tr.lm.interract.step.eval = eval(tr.lm.interract.step.pred, test.norm$quality, plot=T, title="lm wiht interaction and var selection: ");unlist(tr.lm.interract.step.eval)

scatter.smooth(x = test.norm$quality ,tr.lm.interract.step.pred )


#Ridge Regresion

# Data Partition
set.seed(222)
ind <- sample(2, nrow(cleanRedwine), replace = T, prob = c(0.7, 0.3))
train <- cleanRedwine[ind==1,]
test <- cleanRedwine[ind==2,]

# Custom Control Parameters
custom <- trainControl(method = "repeatedcv",
                       number = 10,
                       repeats = 5,
                       verboseIter = T)



set.seed(1234)
ridge <- train(quality~.,
               data= train.norm,
               method = "glmnet",
               tuneGrid = expand.grid(alpha = 0,
                                      lambda = seq(0.0001,1, length = 5)),
               trControl = custom)
plot(ridge)
plot(ridge$finalModel, xvar = "lambda", label = T)
plot(ridge$finalModel, xvar = "dev" , label = T)
plot(varImp(ridge, scale = T))

tr.rr.interract.step.pred = predict(ridge, test.norm[,-12])
tr.rr.interract.step.eval = eval(tr.rr.interract.step.pred, test.norm$quality, plot=T, title="lm wiht interaction and var selection: ");unlist(tr.lm.interract.step.eval)


#Predict


p1<- predict(ridge,train)
sqrt(mean((train$quality-p1)^2))
p2<- predict(ridge,test)
sqrt(mean((test$quality-p2)^2))


predtest2<-data.frame(p2)
final_data1<- cbind(test,predtest2)
write.csv(final_data,"winequality_ridge.csv")

# Regression Tree

tr.rpart = rpart(quality~., data=train.norm)
summary(tr.rpart)
rpart.plot(tr.rpart) 


tr.rpart.pred = predict(tr.rpart, test.norm[,-12])
tr.rpart.eval = eval(tr.rpart.pred, test.norm$quality, plot=T, title = "Regression Tree:"); unlist(tr.rpart.eval)


scatter.smooth(x = test.norm$quality ,tr.rpart.pred )



# Support Vector Machine

tr.svm = ksvm(quality ~ ., 
              data = train.norm, 
              scaled = F,
              kernel = "rbfdot", 
              C = 1)
tr.svm.pred = predict(tr.svm, test.norm[,-12])
tr.svm.eval = eval(tr.svm.pred, test.norm$quality, plot = T, title = "SVM: "); unlist(tr.svm.eval)

scatter.smooth(x = test.norm$quality ,tr.svm.pred)


# Similarly, I will use the train function to optimize model hyperparameters. This step will be computationally expensive.

cl = makePSOCKcluster(4)
registerDoParallel(cl)
tr = trainControl(method = "repeatedcv", number = 10, repeats = 2)
set.seed(1)
tr.svmRadial = train(quality ~.,
                     data = train.norm,
                     method = "svmRadial",
                     trControl=tr,
                     preProcess = NULL,
                     tuneLength = 10)
stopCluster(cl)
save(tr.svmRadial, file="C:/Users/VSBAG/Desktop/DSE_Milan/3rd_sem_subject/Stat/Group_Project/wine_train_cv_svmRadial.RData")
load(file = "C:/Users/VSBAG/Desktop/DSE_Milan/3rd_sem_subject/Stat/Group_Project/wine_train_cv_svmRadial.RData"); tr.svmRadial

#Tuning parameter 'sigma' was held constant at a value of 0.07733002
#RMSE was used to select the optimal model using the smallest value.
#The final values used for the model were sigma = 0.07733002 and C = 1.


tr.svmRadial.pred = predict(tr.svmRadial, test.norm[,-12])
tr.svmRadial.eval = eval(tr.svmRadial.pred, test.norm$quality, plot = T, title = "SVM with CV: "); unlist(tr.svmRadial.eval)

scatter.smooth(x = test.norm$quality ,tr.svmRadial.pred)





# Compare all model

knitr::kable(cbind(lm = unlist(tr.lm.eval),
                   rr = unlist(tr.rr.interract.step.eval),
                   quadratic = unlist(tr.qm.eval),
                   rt = unlist(tr.rpart.eval),
                   it = unlist(tr.lm.interract.step.eval),
                   svm = unlist(tr.svm.eval),
                   svm.cv = unlist(tr.svmRadial.eval)) %>% round(3),
             caption = "ALL MODEL COMPARION TABLE")










# classification

library(ROSE)
library(caret)
library(rpart)
library(ggcorrplot)
library(InformationValue)
library(randomForest)


data <- read.csv("https://raw.githubusercontent.com/vicky61992/StatisticProject_winequality/main/winequality-red.csv", sep = ";", header = T)


#Format outcome variable
data$quality <- ifelse(data$quality >= 7, 1, 0)
data$quality <- factor(data$quality, levels = c(0, 1))


#Univariate analysis
#Dependent variable
#Frequency plot
par(mfrow=c(1,1))
barplot(table(data[[12]]), 
        main = sprintf('Frequency plot of the variable: %s', 
                       colnames(data[12])),
        xlab = colnames(data[12]),
        ylab = 'Frequency')
#Check class BIAS
table(data$quality)
round(prop.table((table(data$quality))),2)



#Independent variable
#Boxplots
par(mfrow=c(3,4))
for (i in 1:(length(data)-1)){
  boxplot(x = data[i], 
          horizontal = TRUE, 
          main = sprintf('Boxplot of the variable: %s', 
                         colnames(data[i])),
          xlab = colnames(data[i]))
}
#Histograms
par(mfrow=c(3,4))
for (i in 1:(length(data)-1)){
  hist(x = data[[i]], 
       main = sprintf('Histogram of the variable: %s',
                      colnames(data[i])), 
       xlab = colnames(data[i]))
}




#Bivariate analysis
#Correlation matrix
ggcorrplot(round(cor(data[-12]), 2), 
           type = "lower", 
           lab = TRUE, 
           title = 
             'Correlation matrix of the red wine quality dataset')



#Outliers
#Identifing outliers
is_outlier <- function(x) {
  return(x < quantile(x, 0.25) - 1.5 * IQR(x) | 
           x > quantile(x, 0.75) + 1.5 * IQR(x))
}
outlier <- data.frame(variable = character(), 
                      sum_outliers = integer(),
                      stringsAsFactors=FALSE)
for (j in 1:(length(data)-1)){
  variable <- colnames(data[j])
  for (i in data[j]){
    sum_outliers <- sum(is_outlier(i))
  }
  row <- data.frame(variable,sum_outliers)
  outlier <- rbind(outlier, row)
}



#Identifying the percentage of outliers
for (i in 1:nrow(outlier)){
  if (outlier[i,2]/nrow(data) * 100 >= 5){
    print(paste(outlier[i,1], 
                '=', 
                round(outlier[i,2]/nrow(data) * 100, digits = 2),
                '%'))
  }
}

#Inputting outlier values
for (i in 4:5){
  for (j in 1:nrow(data)){
    if (data[[j, i]] > as.numeric(quantile(data[[i]], 0.75) + 
                                  1.5 * IQR(data[[i]]))){
      if (i == 4){
        data[[j, i]] <- round(mean(data[[i]]), digits = 2)
      } else{
        data[[j, i]] <- round(mean(data[[i]]), digits = 3)
      }
    }
  }
}


#Splitting the dataset into the Training set and Test set
#Stratified sample
data_ones <- data[which(data$quality == 1), ]
data_zeros <- data[which(data$quality == 0), ]
#Train data
set.seed(123)
train_ones_rows <- sample(1:nrow(data_ones), 0.8*nrow(data_ones))
train_zeros_rows <- sample(1:nrow(data_zeros), 0.8*nrow(data_ones))
train_ones <- data_ones[train_ones_rows, ]  
train_zeros <- data_zeros[train_zeros_rows, ]
train_set <- rbind(train_ones, train_zeros)
table(train_set$quality)
#Test Data
test_ones <- data_ones[-train_ones_rows, ]
test_zeros <- data_zeros[-train_zeros_rows, ]
test_set <- rbind(test_ones, test_zeros)
table(test_set$quality)

#Logistic Regression
lr = glm(formula = quality ~.,
         data = train_set,
         family = binomial)
#Predictions
prob_pred = predict(lr, 
                    type = 'response', 
                    newdata = test_set[-12])
optCutOff <- optimalCutoff(test_set$quality, prob_pred)[1]
y_pred = ifelse(prob_pred > optCutOff, 1, 0)

#Making the confusion matrix
cm_lr = table(test_set[, 12], y_pred)
cm_lr
#Accuracy
accuracy_lr = (cm_lr[1,1] + cm_lr[1,1])/
  (cm_lr[1,1] + cm_lr[1,1] + cm_lr[2,1] + cm_lr[1,2])
accuracy_lr

#ROC curve
par(mfrow = c(1, 1))
roc.curve(test_set$quality, y_pred)



#Decision Tree
dt = rpart(formula = quality ~ .,
           data = train_set,
           method = 'class')
#Predictions
y_pred = predict(dt, 
                 type = 'class', 
                 newdata = test_set[-12])



#Making the confusion matrix
cm_dt = table(test_set[, 12], y_pred)
cm_dt
#Accuracy
accuracy_dt = (cm_dt[1,1] + cm_dt[1,1])/
  (cm_dt[1,1] + cm_dt[1,1] + cm_dt[2,1] + cm_dt[1,2])
accuracy_dt

#ROC curve
roc.curve(test_set$quality, y_pred)



#Random forest

rf = randomForest(x = train_set[-12],
                  y = train_set$quality,
                  ntree = 10)
#Predictions
y_pred = predict(rf, 
                 type = 'class', 
                 newdata = test_set[-12])




#Making the confusion matrix
cm_rf = table(test_set[, 12], y_pred)
cm_rf
#Accuracy
accuracy_rf = (cm_rf[1,1] + cm_rf[1,1])/
  (cm_rf[1,1] + cm_rf[1,1] + cm_rf[2,1] + cm_rf[1,2])
accuracy_rf

#ROC curve

roc.curve(test_set$quality, y_pred)


#Variable importance
varImp(lr)

# Compare all model

knitr::kable(cbind(lr = unlist(accuracy_lr),
                   dt = unlist(accuracy_dt),
                   rf = unlist(accuracy_rf)) %>% round(3),
             caption = "ALL MODEL COMPARION TABLE")



































