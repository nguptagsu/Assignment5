#Importing the required packages
library(caret)
library(gbm)
library(dplyr)
library(RANN)

#Importing the dataset
data(scat)

###1. Set the Species column as the target/outcome and convert it to numeric.
colnames(scat)[1] = "outcome"
scat$outcome = as.numeric(as.factor(scat$outcome))


#2. Remove the Month, Year, Site, Location features.
df = subset(scat, select = -c(Month,Year, Site, Location))


#3. Check if any values are null. If there are, impute missing values using KNN
sum(is.na(df))
#Imputing missing values using KNN.Also centering and scaling numerical columns
impute <- preProcess(df, method = c("knnImpute","center","scale"))
newdf <- predict(impute, df)
sum(is.na(newdf))

#4. Converting every categorical variable to numerical (if needed).
#The categorical variables are already numeric.


#Converting the last four columns which only had 0 and 1 values to their original state
newdf$scrape <- df$scrape
newdf$flat <- df$flat
newdf$segmented <- df$segmented
newdf$ropey <- df$ropey
newdf$outcome <- df$outcome

#Converting outcome to categorical
newdf$outcome <- as.factor(newdf$outcome)

#5. With a seed of 100, 75% training, 25% testing .Build the following models: randomforest,
#neuralnet, naive bayes and GBM.
#Spliting training set into two parts based on outcome: 75% and 25%
set.seed(100)
index <- createDataPartition(newdf$outcome, p=0.75, list=FALSE)
trainSet <- newdf[ index,]
testSet <- newdf[-index,]

predictors<-c("Age", "Number", "Length", "Diameter", "Taper", "TI", "Mass", "d13C", "d15N", "CN", "ropey", "segmented", "flat", "scrape")
outcomeName <- "outcome"

#####Models#####
##GBM##
#Model
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm')
#Summary
print(model_gbm)
#Plot
plot(varImp(object=model_gbm),main="GBM - Variable Importance")
#Predictions
predictions_gbm<-predict.train(object=model_gbm,testSet[,predictors],type="raw")
table(predictions_gbm)
#Confusion Matrix and Statistics
confusionMatrix(predictions_gbm,testSet[,outcomeName])


##Random Forest##
#Model
model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf')
#Summary
print(model_rf)
#Plot
plot(varImp(object=model_rf),main="RF - Variable Importance")
#Predictions
predictions_rf<-predict.train(object=model_rf,testSet[,predictors],type="raw")
table(predictions_rf)
#Confusion Matrix and Statistics
confusionMatrix(predictions_rf,testSet[,outcomeName])


##Neural Net##
#Model
model_nnet<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet', importance=T)
#Summary
print(model_nnet)
#Plot
n <- varImp(object=model_nnet)
n$importance <- data.frame(n$importance[,1])
plot(n, main="nnet - Variable Importance")
#Predictions
predictions_nnet<-predict.train(object=model_nnet,testSet[,predictors],type="raw")
table(predictions_nnet)
#Confusion Matrix and Statistics
confusionMatrix(predictions_nnet,testSet[,outcomeName])


##Naive Bayes##
#Model
model_naive_bayes<-train(trainSet[,predictors],trainSet[,outcomeName],method='naive_bayes')
#Summary
print(model_naive_bayes)
#Plot
plot(varImp(object=model_naive_bayes),main="naive_bayes - Variable Importance")
#Predictions
predictions_naive_bayes<-predict.train(object=model_naive_bayes,testSet[,predictors],type="raw")
table(predictions_naive_bayes)
#Confusion Matrix and Statistics
confusionMatrix(predictions_naive_bayes,testSet[,outcomeName])

#6. For the BEST performing models of each (randomforest, neural net, naive bayes and gbm) create
#and display a data frame that has the following columns: ExperimentName, accuracy, kappa.
#Sort the data frame by accuracy.

top_gbm <- c(model_gbm$method, model_gbm$results[1, "Accuracy"], model_gbm$results[1, "Kappa"])
top_rf <- c(model_rf$method, model_rf$results[1, "Accuracy"], model_rf$results[1, "Kappa"])
top_nnet <- c(model_nnet$method, model_nnet$results[1, "Accuracy"], model_nnet$results[1, "Kappa"])
top_naive_bayes <- c(model_naive_bayes$method, model_naive_bayes$results[1, "Accuracy"], model_naive_bayes$results[1, "Kappa"])

consolidated = list(top_gbm, top_rf, top_nnet, top_naive_bayes)
top <- data.frame(matrix(unlist(consolidated), nrow=4, byrow=T))
colnames(top)[1] = "ExperimentName"
colnames(top)[2] = "accuracy"
colnames(top)[3] = "kappa"
top <- top[order(top$accuracy, decreasing = TRUE), ]
top

##############################################################################
#7. Tune the GBM model using tune length = 20 and: a) print the model summary and b) plot the
#models.
### Using tuneLength ###

#using tune length
model_gbm_tune_length<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',tuneLength=20)
print(model_gbm_tune_length)

# visualize the models
plot(model_gbm_tune_length)

##################################################
#8. Using GGplot and gridExtra to plot all variable of importance plots into one single plot.
one <- ggplot(varImp(object=model_gbm)) + ggtitle("GBM - Variable Importance")
two <- ggplot(varImp(object=model_rf)) + ggtitle("RF - Variable Importance")
three <- ggplot(n) + ggtitle("nnet - Variable Importance")
four <- ggplot(varImp(object=model_naive_bayes)) + ggtitle("naive_bayes - Variable Importance")

grid.arrange(one,two, three, four, nrow = 2, ncol = 2)

#########################################################
#9.Which model performs the best? and why do you think this is the case? Can we accurately
#predict species on this dataset?
#Out of the non-tuned models, gradient boosting performed the best with accuracy near to 70%.
#I think this is the case due to its capability to convert weak learners into strong learners. 
#We can predict the species with 70% accuracy.
#*******************************************##########****************************************************************
#######################################################
#10-A Using feature selection with rfe in caret and the repeatedcv method: Find the top 3
#predictors and build the same models as in 6 and 8 with the same parameters.
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)
top3<-names(trainSet)[!names(trainSet) %in% outcomeName]
columns <- rfe(trainSet[,top3], trainSet[,outcomeName],
               rfeControl = control)
columns

top3 <- c("d15N", "Mass", "d13C")

#####Models#####
##GBM##
#Model
model_top3_gbm<-train(trainSet[,top3],trainSet[,outcomeName],method='gbm')
#Summary
print(model_top3_gbm)
#Plot
plot(varImp(object=model_top3_gbm),main="GBM - Variable Importance")
#Predictions
predictions_gbm_top3<-predict.train(object=model_top3_gbm,testSet[,top3],type="raw")
table(predictions_gbm_top3)
#Confusion Matrix and Statistics
confusionMatrix(predictions_gbm_top3,testSet[,outcomeName])


##Random Forest##
#Model
model_top3_rf<-train(trainSet[,top3],trainSet[,outcomeName],method='rf')
#Summary
print(model_top3_rf)
#Plot
plot(varImp(object=model_top3_rf),main="RF - Variable Importance")
#Predictions
predictions_rf_top3<-predict.train(object=model_top3_rf,testSet[,top3],type="raw")
table(predictions_rf_top3)
#Confusion Matrix and Statistics
confusionMatrix(predictions_rf_top3,testSet[,outcomeName])


##Neural Net##
#Model
model_top3_nnet<-train(trainSet[,top3],trainSet[,outcomeName],method='nnet', importance=T)
#Summary
print(model_top3_nnet)
#Plot
nt <- varImp(object=model_top3_nnet)
nt$importance <- data.frame(nt$importance[,1])
plot(nt, main="nnet - Variable Importance")
#Predictions
predictions_nnet_top3<-predict.train(object=model_top3_nnet,testSet[,top3],type="raw")
table(predictions_nnet_top3)
#Confusion Matrix and Statistics
confusionMatrix(predictions_nnet_top3,testSet[,outcomeName])


##Naive Bayes##
#Model
model_top3_naive_bayes<-train(trainSet[,top3],trainSet[,outcomeName],method='naive_bayes')
#Summary
print(model_top3_naive_bayes)
#Plot
plot(varImp(object=model_top3_naive_bayes),main="naive_bayes - Variable Importance")
#Predictions
predictions_naive_bayes_top3<-predict.train(object=model_top3_naive_bayes,testSet[,top3],type="raw")
table(predictions_naive_bayes_top3)
#Confusion Matrix and Statistics
confusionMatrix(predictions_naive_bayes_top3,testSet[,outcomeName])

top3_gbm <- c(model_top3_gbm$method, model_top3_gbm$results[1, "Accuracy"], model_top3_gbm$results[1, "Kappa"])
top3_rf <- c(model_top3_rf$method, model_top3_rf$results[1, "Accuracy"], model_top3_rf$results[1, "Kappa"])
top3_nnet <- c(model_top3_nnet$method, model_top3_nnet$results[1, "Accuracy"], model_top3_nnet$results[1, "Kappa"])
top3_naive_bayes <- c(model_top3_naive_bayes$method, model_top3_naive_bayes$results[1, "Accuracy"], model_top3_naive_bayes$results[1, "Kappa"])

top3_naive_bayes[1] <- "Naive Bayes Top3"
top3_nnet[1] <- "Neural-Net Top3"
top3_rf[1] <- "Random Forest Top3"
top3_gbm[1] <- "GBM Tuned Top3"

consolidated3 = list(top3_gbm, top3_rf, top3_nnet, top3_naive_bayes)
top3df <- data.frame(matrix(unlist(consolidated3), nrow=4, byrow=T))
colnames(top3df)[1] = "ExperimentName"
colnames(top3df)[2] = "accuracy"
colnames(top3df)[3] = "kappa"
top3df <- top3df[order(top3df$accuracy, decreasing = TRUE), ]
top3df

a <- ggplot(varImp(object=model_top3_gbm)) + ggtitle("GBM Top3 - Variable Importance")
b <- ggplot(varImp(object=model_top3_rf)) + ggtitle("RF Top3 - Variable Importance")
c <- ggplot(nt) + ggtitle("NNET top3 - Variable Importance")
d <- ggplot(varImp(object=model_top3_naive_bayes)) + ggtitle("naive_bayes top3 - Variable Importance")

grid.arrange(a,b, c, d, nrow = 2, ncol = 2)

##############################################################
#10-B Create a dataframe that compares the non-feature selected models ( the same as on 7)
#and add the best BEST performing models of each (randomforest, neural net, naive
#bayes and gbm) and display the data frame that has the following columns:
#ExperimentName, accuracy, kappa. Sort the data frame by accuracy.
#Tuning the models with tune length 20
model_gbm_tune_length<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',tuneLength=20)
print(model_gbm_tune_length)
gbm_tune <- c(model_gbm_tune_length$method, model_gbm_tune_length$results[1, "Accuracy"], model_gbm_tune_length$results[1, "Kappa"])

model_nb_tune_length<-train(trainSet[,predictors],trainSet[,outcomeName],method='naive_bayes',tuneLength=20)
print(model_nb_tune_length)
nb_tune <- c(model_nb_tune_length$method, model_nb_tune_length$results[1, "Accuracy"], model_nb_tune_length$results[1, "Kappa"])

model_nnet_tune_length<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet',tuneLength=20)
print(model_nnet_tune_length)
nnet_tune <- c(model_nnet_tune_length$method, model_nnet_tune_length$results[1, "Accuracy"], model_nb_tune_length$results[1, "Kappa"])

model_rf_tune_length<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf',tuneLength=20)
print(model_rf_tune_length)
rf_tune <- c(model_rf_tune_length$method, model_rf_tune_length$results[1, "Accuracy"], model_nb_tune_length$results[1, "Kappa"])

nb_tune[1] <- "Naive Bayes Tuned"
nnet_tune[1] <- "Neural-Net Tuned"
rf_tune[1] <- "Random Forest Tuned"
gbm_tune[1] <- "GBM Tuned"

consolidatedAll = list(gbm_tune, nb_tune, nnet_tune, rf_tune)
all <- data.frame(matrix(unlist(consolidatedAll), nrow=4, byrow=T))
colnames(all)[1] = "ExperimentName"
colnames(all)[2] = "accuracy"
colnames(all)[3] = "kappa"
all <- all[order(all$accuracy, decreasing = TRUE), ]
all

#Also including the commented code to generate a dataframe which compares top3 and tuned
#models
#Consolidatng all the model accuracies
#consolidatedAll = list(gbm_tune, nb_tune, nnet_tune, rf_tune, top3_naive_bayes, top3_nnet, top3_rf, top3_gbm)
#all <- data.frame(matrix(unlist(consolidatedAll), nrow=8, byrow=T))
#colnames(all)[1] = "ExperimentName"
#colnames(all)[2] = "accuracy"
#colnames(all)[3] = "kappa"
#all <- all[order(all$accuracy, decreasing = TRUE), ]
#all

##############################################################
#10-C Which model performs the best? and why do you think this is the case? Can we
#accurately predict species on this dataset?
#The GBM tuned model performs the best among all. 
#I think the reason for this is ‘Boosting’. 
#The GBM model converts the weak learners into strong learners. 
#We can 70% accurately predict the species on this dataset.
