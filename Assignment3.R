setwd("/Users/ankurkhanna/Desktop/Ml and Data Mining/first")
train <- read.csv("train.csv",header = T)
test <- read.csv("test.csv", header = T)

# Combining data
titanic <- bind_rows(train, test)

# Checking the structure of the data
str(titanic)

titanic$Pclass = as.factor(titanic$Pclass)
#titanic$Embarked = as.factor(titanic$Embarked)

# Checking missing values 
summary(titanic)
colSums(is.na(titanic)|titanic=='')

#Cabin has the most number of missing values, 1014 values. Age has 263 missing values while Embarked and Fare have two and one missing values, respectively

#Imputing mean value of Fare in missing calues
fare = c(titanic$Fare)
fare = mean(fare, na.rm = T)
titanic$Fare[is.na(titanic$Fare)] = fare

#Imputing missing value of Embarked with mode
table(titanic$Embarked)
titanic[titanic$Embarked=='', "Embarked"] <- 'S'

#Imputing missing value of Age as per Pclass
by(titanic$Age,titanic$Sex, summary)
by(titanic$Age,titanic$Pclass, summary)
#function to compute missing values of age with mean as per class using if-else
compute.age <- function(age,class){
  vector <- age
  for (i in 1:length(age)){
    if (is.na(age[i])){
      if (class[i] == 1){
        vector[i] <- round(mean(filter(titanic,Pclass==1)$Age, na.rm=TRUE),0)
      }else if (class[i] == 2){
        vector[i] <- round(mean(filter(titanic,Pclass==2)$Age, na.rm=TRUE),0)
      }else{
        vector[i] <- round(mean(filter(titanic,Pclass==3)$Age, na.rm=TRUE),0)
      }
    }else{
      vector[i]<-age[i]
    }
  }
  return(vector)
}
compute.age <- compute.age(titanic$Age,titanic$Pclass)
titanic$Age <- compute.age

titanic$Survived = factor(titanic$Survived)
titanic$Pclass = factor(titanic$Pclass)
titanic$Sex = factor(titanic$Sex)
titanic$Embarked = factor(titanic$Embarked)
#Checking the structure of the data
str(titanic)

# Splitting the dataset into the Training set and Test set
train_split <- titanic[1:891, c("Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked")]
test_split <- titanic[892:1309, c("Pclass","Sex","Age","SibSp","Parch","Fare","Embarked")]

# Splitting the Training set into the Training set and Validation set
library(caTools)
split = sample.split(train_split$Survived, SplitRatio = 0.8)
trainingset = subset(train_split, split == TRUE)
validationset = subset(train_split, split == FALSE)

########################
#Logistic Regression
logmodel = glm(Survived ~., family = binomial(link='logit'), data = trainingset)
summary(logmodel)
# Choosing the best model by AIC in a Stepwise Algorithm
# The step() function iteratively removes insignificant features from the model.
logmodel <- step(logmodel)

# Predicting the Validation set results
logpred = predict(logmodel, type = 'response', newdata = validationset)
y_pred = ifelse(logpred > 0.5, 1, 0)
# Checking the prediction accuracy
table(validationset$Survived, y_pred > 0.5) # Confusion matrix
#checking accuracy of Validation set
error <- mean(validationset$Survived != y_pred) # Misclassification error
paste('Accuracy',round(1-error,4))

#checking accuracy of test set
prediction = predict(logmodel, type = 'response',newdata = test_split)
View(prediction)
prediction = ifelse(prediction >0.5, 1, 0)
solution <- data.frame(PassengerID = titanic[892:1309,"PassengerId"], Survived = prediction)
#solution <- data.frame(PassengerID = test_split$PassengerId, Survived = prediction)
write.csv(solution, file = 'Logistic.csv', row.names = F)
#75.5% accuracy on Kaggle

#######################
#Random forest
library(randomForest)
rfmodel = randomForest(Survived ~., data = trainingset)
plot(rfmodel) 

# Predicting the Validation set results
y_pred = predict(rfmodel, newdata = validationset[,-which(names(validationset)=="Survived")])

# Checking the prediction accuracy
table(validationset$Survived, y_pred) # Confusion matrix
# Misclassification error
error <- mean(validationset$Survived != y_pred) # Misclassification error
paste('Accuracy',round(1-error,4))

#checking accuracy of test set
prediction = predict(rfmodel, type = 'response',newdata = test_split)
View(prediction)
solution <- data.frame(PassengerID = titanic[892:1309,"PassengerId"], Survived = prediction)
#solution <- data.frame(PassengerID = test_split$PassengerId, Survived = prediction)
write.csv(solution, file = 'rfmodel.csv', row.names = F)
#76.5% accuracy on Kaggle


###########################
#Naive Bayes
# Fitting Naive Bayes to the Training set
Bayesmodel = naiveBayes(Survived ~ Sex + Age + Pclass, data = trainingset)

# Predicting the Validation set results
y_pred = predict(Bayesmodel, newdata = validationset[,-which(names(validationset)=="Survived")])

# Checking the prediction accuracy
table(validationset$Survived, y_pred) # Confusion matrix

error <- mean(validationset$Survived != y_pred) # Misclassification error
paste('Accuracy',round(1-error,4))

#checking accuracy of test set
prediction = predict(Bayesmodel,newdata = test_split)
View(prediction)
solution <- data.frame(PassengerID = titanic[892:1309,"PassengerId"], Survived = prediction)
#solution <- data.frame(PassengerID = test_split$PassengerId, Survived = prediction)
write.csv(solution, file = 'Bayes.csv', row.names = F)

#76.5% accuarcy on kaggle

#####################
#SVM model



library(caret)
mSVM <- train(as.factor(Survived)~. ,data=trainingset,method = 'svmRadial',trControl=trainControl(method='cv',number=10))
confusionMatrix(mSVM, positive = '1')
y_pred <- predict(mSVM, newdata = test_split)
solution<-data.frame(PassengerID = titanic[892:1309,"PassengerId"], Survived = y_pred)
write.csv(solution, file = 'svm.csv', row.names = F)
#79% accuracy on kaggle
mSVM2 <- train(as.factor(Survived)~. ,data=trainingset,method = 'svmLinear',trControl=trainControl(method='cv',number=10))
mSVM3 <- train(as.factor(Survived)~. ,data=trainingset,method = 'svmPoly',trControl=trainControl(method='cv',number=10))
confusionMatrix(mSVM2, positive = '1')
confusionMatrix(mSVM3, positive = '1')
y_pred1 <- predict(mSVM2, newdata = test_split)
y_pred2 <- predict(mSVM3, newdata = test_split)
solution1<-data.frame(PassengerID = titanic[892:1309,"PassengerId"], Survived = y_pred1)
solution2<-data.frame(PassengerID = titanic[892:1309,"PassengerId"], Survived = y_pred2)
write.csv(solution1, file = 'svm2.csv', row.names = F) #77% accuracy in kaggle
write.csv(solution2, file = 'svm3.csv', row.names = F) #78.4% accuarcy in kaggle

