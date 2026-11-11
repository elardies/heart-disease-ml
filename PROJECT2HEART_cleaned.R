# Heart Disease Prediction — Statistical Learning (reconstructed)
# Cleaned for portability; original analysis and feature-selection methods preserved.
# Place heart.csv in the RStudio working directory before running.
# Required packages (install once if missing):

required_packages <- c(
  "dplyr", "tidyr", "ggplot2", "GGally", "e1071", "VIM",
  "caret", "pROC", "glmnet", "MASS", "rpart", "rpart.plot",
  "randomForest", "class"
)
missing_packages <- setdiff(required_packages, rownames(installed.packages()))
if (length(missing_packages) > 0) {
  stop("Install missing packages first: install.packages(c(",
       paste(sprintf('"%s"', missing_packages), collapse = ", "), "))")
}
invisible(lapply(required_packages, function(pkg) {
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}))

if (!file.exists("heart.csv")) {
  stop("heart.csv not found. Set the RStudio working directory to the folder containing heart.csv.")
}
readLines("heart.csv", 3)
data_original <- read.csv("heart.csv", header = TRUE, sep = ",", dec = ".")

#Deal with missing values
table(is.na(data_original)) 

colSums(is.na(data_original))

#Only 4 missing values, we can delete those rows.
data = na.omit(data_original)

#We realize that there are many 0 values in Cholesterol, which is it impossible, 
#so we deal with them as missing values
sum(data$Cholesterol == 0)


#We replaced the 0 values with NA. 
data$Cholesterol <- replace(data$Cholesterol,
                            data$Cholesterol == 0,
                            NA)

#We need to look at it's skewness
skewness_Cholesterol = skewness(data$Cholesterol, na.rm = TRUE) 
print(skewness_Cholesterol)
hist(data$Cholesterol, main = "Histogram of Cholesterol", xlab = "Value", col = "skyblue", breaks = 30) 

#It is clearly skewness to the right. To deal with it we use knn.
df_skewed_imputed <- kNN(
  data     = data,          
  variable = "Cholesterol", 
  k        = 5
)

summary(df_skewed_imputed$Cholesterol)

#Check if there are any NA
missing_values_imputed = sapply(df_skewed_imputed, function(x) sum(is.na(x))) 
print(missing_values_imputed)

#Change the original variables for the ones without NA that we had computed using knn  
data$Cholesterol = df_skewed_imputed$Cholesterol



#Factorize categorical variables
data$HeartDisease = as.factor(data$HeartDisease) 
data$HeartDisease = factor(data$HeartDisease, levels = c(0, 1), 
                      labels = c("No", "Yes"))

data$Sex           = as.factor(data$Sex)
data$ChestPainType = as.factor(data$ChestPainType)
data$FastingBS     = as.factor(data$FastingBS)
data$FastingBS = factor(data$FastingBS, levels = c(0, 1), 
                           labels = c("No", "Yes"))

data$RestingECG    = as.factor(data$RestingECG)
data$ExerciseAngina = as.factor(data$ExerciseAngina)
data$ST_Slope      = as.factor(data$ST_Slope)




#Outliers
num_vars = data[, c("RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "Age")]
#Visualize outliers with using boxplots
boxplot(num_vars, 
        las=2, 
        col="grey",
        main = "Outliers of Numeric Variables",
        outcol = "red")


outliers= numeric(ncol(num_vars))
# Loop through each variable to to calculate IQR and identify the outliers
for (i in 1:length(num_vars)) {
  # Calculate Q1 and Q3
  QI = quantile(num_vars[[i]], 0.25, na.rm = TRUE) #1st quantile
  QS = quantile(num_vars[[i]], 0.75, na.rm = TRUE) #3rd quantile
  IQR = QS - QI #Interquartile range (IQR)
  # Count outliers
  outliers[i] = sum(num_vars[[i]] < (QI - 1.5 * IQR) | num_vars[[i]] >
                      (QS + 1.5 * IQR), na.rm = TRUE)
}
# Print the number of outliers identified for each variable
names(outliers) = names(num_vars)
print(outliers)



#In this dataset there are several observations that appear as outliers in the 
#numeric variables. However, we decided not to remove all of them, and instead 
#only excluded values that are clearly impossible or inconsistent with basic 
#clinical knowledge. Although many of these points are far from the median, 
#they still fall within plausible physiological ranges. For a study that aims to 
#predict the presence of heart disease, retaining such extreme but realistic values 
#is important, because it is precisely these atypical measurements (for example 
#very high blood pressure or cholesterol) that may be most strongly associated 
#with the disease and therefore contain valuable information for the model.



#Instead of removing the outliers, we have done a research about the possible values
#of these parameters, and we have removed only the imposible values. 

data_clean <- data |>
  dplyr::filter(
    RestingBP >= 60,   RestingBP <= 220,   
    Cholesterol >= 80, Cholesterol <= 700, 
    MaxHR    >= 60,    MaxHR    <= 220,    
    Oldpeak  >= -2.5,  Oldpeak  <= 6       
  )

#values removed


#CORRELATION:Visualize the correlation of variables. We use only the numeric variables to compute the correlation matrix

vars_for_coer = data_clean %>% 
  dplyr::select(RestingBP, Cholesterol, MaxHR, Oldpeak, Age)

cor_matrix = ggcorr(vars_for_coer, label = TRUE, hjust = 0.75,
                    label_size = 4, label_color = "black",
                    low = "turquoise", high = "coral2", mid = "white",
                    midpoint = 0, name = "Correlation") +
  labs(title = "Correlation Matrix")

print(cor_matrix)


#creation of new variables 
data_clean$HighRisk = ifelse(data_clean$RestingBP > 140 & data_clean$Cholesterol > 240, "yes", "no")
table(data$data_clean)

data_clean$HighRiskAngina = ifelse(data_clean$ChestPainType %in% c("ASY","NAP") &
                                     data_clean$ExerciseAngina == "Y", "yes", "no")
table(data_clean$HighRiskAngina)

data_clean$HighRiskST = ifelse(data_clean$ST_Slope == "Down" & data_clean$Oldpeak > 2, "yes", "no")
table(data_clean$HighRiskST)



#Visualization tools
set.seed(123)
inTrain <- createDataPartition(data_clean$HeartDisease,
                               p = 0.8, list = FALSE)
train <- data_clean[inTrain, ]
test  <- data_clean[-inTrain, ]
nrow(train) #here we see how many observations are for training and below for testing 
nrow(test)  
table(train$HeartDisease)/length(train$HeartDisease)


#proportion of heart disease
syndrome_counts <- as.data.frame(table(train$HeartDisease))
colnames(syndrome_counts) <- c("HeartDisease", "Count")

syndrome_counts$Percentage <-
  round(syndrome_counts$Count / sum(syndrome_counts$Count) * 100, 1)

ggplot(syndrome_counts, aes(x = "", y = Count, fill = HeartDisease)) +
  geom_bar(stat = "identity", width = 1, color = "black") +
  coord_polar(theta = "y") +
  ggtitle("Proportion of HeartDisease") +
  theme_void() +
  geom_text(
    aes(label = paste(Count, " (", Percentage, "%)", sep = "")),
    position = position_stack(vjust = 0.5),
    color = "black",
    size = 5
  )

#Cholesterol distribution by HeartDisease
ggplot(train, aes(x = Cholesterol, fill = HeartDisease)) +
  geom_density(alpha = 0.5) +
  ggtitle("Cholesterol distribution by HeartDisease") +
  xlab("Cholesterol")
#The cholesterol distributions for patients with and without heart disease are 
#very similar, with only a slight shift toward higher values in the “Yes” group. 
#This suggests that cholesterol alone is not a strong separator between classes.


#Heart Disease Counts by Chest Pain Type
ggplot(train, aes(x = ChestPainType, fill = HeartDisease)) +
  geom_bar(position = "dodge") +
  ggtitle("Heart Disease Counts by Chest Pain Type") +
  xlab("Chest Pain Type") +
  ylab("Count") +
  theme_minimal()

#The plot shows that the probability of having heart disease varies strongly by 
#chest pain type. Patients with ASY chest pain are mostly in the HeartDisease = “Yes” 
#group, while those with ATA are predominantly “No”, and NAP/TA lie in between, 
#indicating that chest pain type is a very informative predictor of heart disease status.


# RestingBP by HeartDisease status
ggplot(train, aes(x = HeartDisease, y = RestingBP, fill = HeartDisease)) +
  geom_violin(trim = FALSE, alpha = 0.8) +
  ggtitle("RestingBP by HeartDisease status") +
  theme_minimal()

#The distribution of resting blood pressure is very similar between patients with 
#and without heart disease, with no major shift in the central values.
#There are extreme values in both groups, so RestingBP alone does not clearly 
#distinguish the presence of heart disease.



# MaxHR by HeartDisease status
ggplot(train, aes(x = HeartDisease, y = MaxHR, fill = HeartDisease)) +
  geom_boxplot(outlier.color = "red4") +
  ggtitle("MaxHR by HeartDisease status") +
  theme_minimal()
#lower achievable MaxHR is associated with having heart disease, which is 
#consistent with reduced exercise capacity or chronotropic response in cardiac patients.


#Pairwise relationships between key variables
ggpairs(train,columns = c("Age", "Cholesterol", "MaxHR"),  
  aes(color = HeartDisease)
  ) +
  ggtitle("Pairwise relationships between key variables")

#The pairwise plot shows weak correlations among the numeric variables and a 
#negative relationship between Age and MaxHR.


#Age distribution by HeartDisease status
ggplot(train, aes(x = Age, fill = HeartDisease)) +
  geom_density(alpha = 0.5) +
  ggtitle("Age distribution by HeartDisease status") +
  xlab("Age") +
  theme_minimal()
#The age distribution of patients with and without heart disease overlaps a lot, 
#but the peak for HeartDisease = “No” is slightly younger than for HeartDisease = “Yes”.
#Patients with heart disease tend to be somewhat older on average, although there 
#are individuals of many ages in both groups.


#Number of patients by Sex and HeartDisease
ggplot(train, aes(x = Sex, fill = HeartDisease)) +
  geom_bar(position = "dodge") +
  ylab("Count") +
  ggtitle("Number of patients by Sex and HeartDisease") +
  theme_minimal()
#The chart shows that most patients in the dataset are male, and among men there 
#are more cases with heart disease than without.For women, both the total number 
#of patients and the number with heart disease are much smaller, suggesting that 
#heart disease is less frequent in females in this sample.



#HeartDisease proportion by engineered risk variables
risk_long_train <- train |>
  dplyr::select(HeartDisease, HighRisk, HighRiskAngina, HighRiskST) |>
  tidyr::pivot_longer(
    cols = c(HighRisk, HighRiskAngina, HighRiskST),
    names_to = "RiskType",
    values_to = "RiskFlag"
  )

ggplot(risk_long_train, aes(x = RiskFlag, fill = HeartDisease)) +
  geom_bar(position = "fill") +
  facet_wrap(~ RiskType) +
  ylab("Proportion") +
  xlab("RiskFlag") +
  ggtitle("HeartDisease proportion by engineered risk variables") +
  theme_minimal()

#Across all three engineered risk flags (HighRisk, HighRiskAngina and HighRiskST), 
#the “yes” category shows a much higher proportion of patients with heart disease 
#than the “no” category.This means that when any of these risk flags is present, 
#the probability of having heart disease increases substantially compared with 
#patients without that risk flag.




####
# Train / test split has been done already

table(train$HeartDisease)/length(train$HeartDisease)
##Logistic regression
logit.model = glm(HeartDisease~ Age + Sex + ChestPainType + RestingBP +
                    Cholesterol + FastingBS + RestingECG + MaxHR +
                    ExerciseAngina + Oldpeak + ST_Slope, 
                  family = binomial(link = 'logit'), data = train)

summary(logit.model)

#Predicted propabilities for standard logistic regression model
probability = predict(logit.model,newdata=test, type='response')

prediction <- ifelse(probability < 0.5, "No", "Yes")
prediction <- factor(prediction, levels = c("No","Yes"))
confusionMatrix(prediction, test$HeartDisease)

#Accuracy : 0.8736
#Kappa : 0.7445 

# ROC curve and AUC for standard logistic regression
model = logit.model
probability = predict(model, test, type = "response")
roc.glm = roc(test$HeartDisease, probability)
auc(roc.glm)

#Plot results 
plot.roc(test$HeartDisease, probability,col="hotpink", print.auc = TRUE,
         auc.polygon=TRUE, grid=c(0.1, 0.2),
         grid.col=c("purple", "violet"), max.auc.polygon=TRUE,
         auc.polygon.col="lightyellow", print.thres=TRUE, legacy.axes = TRUE)

#The standard logistic regression model shows excellent predictive performance, 
#with an AUC of ≈ 0.94.The ROC curve lies close to the top-left corner, 
#reflecting high sensitivity and specificity across a wide range of thresholds. 
#Using the optimal cut-off, the model achieves a good balance between correctly 
#identifying diseased patients and minimizing false positives.


##Penalized Logistic regression
ctrl = trainControl(method = "cv", number = 5,
                    classProbs = TRUE,
                    verboseIter=T)
lrFit = train(HeartDisease~ Age + Sex + ChestPainType + RestingBP +
                Cholesterol + FastingBS + RestingECG + MaxHR +
                ExerciseAngina + Oldpeak + ST_Slope,
              method = "glmnet",
              tuneGrid = expand.grid(alpha = seq(0, 1, 0.1), lambda = seq(0, .1, 0.02)),
              metric = "Kappa",
              data = train,
              preProcess = c("center", "scale"),
              trControl = ctrl)

#best alpha/lambda and CV performance
print(lrFit)
#alpha = 0.2, lambda = 0.06 

#Accuracy : 0.8668871 
#Kappa : 0.7288381

# Variable importance from the penalized model
lr_imp = varImp(lrFit, scale = F)
plot(lr_imp, scales = list(y = list(cex = .95)))

#Predicted probabilities for penalized model
lrProb = predict(lrFit, test, type="prob")

# ROC curve and AUC for penalized logistic regression
roc.lr <- roc(test$HeartDisease, lrProb$Yes)
auc(roc.lr)

#Plot results 
plot.roc(test$HeartDisease, lrProb$Yes, col="blue", print.auc=TRUE,
         auc.polygon=TRUE, grid=c(0.1,0.2),
         grid.col=c("blue","turquoise"), max.auc.polygon=TRUE,
         auc.polygon.col="lightyellow", print.thres=TRUE, legacy.axes=TRUE)


#Penalized logistic regression achieves almost the same performance, with an AUC 
#also around 0.94.By applying regularization, the model shrinks less informative 
#coefficients toward zero and highlights the most important predictors, 
#improving stability and reducing potential overfitting.

##LDA (Linear Discriminant Analysis)
#For continuous variables
lda.model=lda(HeartDisease~Age + RestingBP + Cholesterol + MaxHR + Oldpeak, data=train)
summary(lda.model)

#Predicted probabilities for LDA model
lda.prediction = predict(lda.model, newdata = test)
lda.probabilities = lda.prediction$posterior[, 2]
predicted.classes.lda = lda.prediction$class

conf_matrix_lda = confusionMatrix(predicted.classes.lda, test$HeartDisease)
conf_matrix_lda

#Accuracy : 0.7473
#Kappa : 0.4871 

# ROC curve and AUC for penalized logistic regression
roc_curve_lda = roc(test$HeartDisease, lda.probabilities)
auc_value_lda = auc(roc_curve_lda)
print(auc_value_lda)

#Plot results
plot.roc(test$HeartDisease, lda.probabilities,col="purple", print.auc = TRUE,
         auc.polygon=TRUE, grid=c(0.1, 0.2),
         grid.col=c("darkgreen", "darkgreen"), max.auc.polygon=TRUE,
         auc.polygon.col="lightpink", print.thres=TRUE, legacy.axes = TRUE)


#LDA using only continuous predictors (Age, RestingBP, Cholesterol, MaxHR, 
#Oldpeak) yields an AUC ≈ 0.85, indicating good but clearly inferior performance 
#compared with the full logistic regression (AUC ≈ 0.94). The ROC curve shows 
#that at an optimal threshold around 0.63, LDA attains high sensitivity (~0.90) 
#but more moderate specificity (~0.67).

##QDA (Quadratic Discriminant Analysis)
#assumes that the data within each class follows a normal distribution, but unlike 
#LDA, it allows each class to have its own covariance matrix.

#(continuous variables)
qda.model = qda(HeartDisease~Age + RestingBP + Cholesterol + MaxHR + Oldpeak, data=train)
summary(qda.model)

# Predict probabilities for QDA model
qda.prediction = predict(qda.model, newdata = test)

predicted.classes.qda = qda.prediction$class
actual.classes.qda = test$HeartDisease

kappa_value = confusionMatrix(predicted.classes.qda, actual.classes.qda)$overall['Kappa']
print(kappa_value)

qda.probs = qda.prediction$posterior[, 2]


conf_matrix = table(predicted.classes.qda, test$HeartDisease)
print(conf_matrix)

#Accuracy : 0.75824
#Kappa : 0.51177

# ROC curve and AUC for QDA
roc_curve_qda = roc(test$HeartDisease, qda.probs)
auc(roc_curve_qda)

#Plot results
plot.roc(test$HeartDisease, qda.probs,
         col = "darkgreen",
         print.auc = TRUE,
         auc.polygon = TRUE,
         grid = c(0.1, 0.2),
         grid.col = c("purple", "purple"),
         max.auc.polygon = TRUE,
         auc.polygon.col = "palegreen",
         print.thres = TRUE,
         legacy.axes = TRUE)

#A QDA model trained on the continuous predictors produced an AUC of 0.83, 
#showing good but lower discriminative power than logistic regression and LDA. 
#The optimal threshold (≈0.61) yields high sensitivity (0.88) but moderate 
#specificity (0.68). QDA’s flexibility helps capture nonlinear patterns, though 
#it may overfit when sample sizes are limited.

##Naive Bayes
#Assumes independence between variables given the class and uses Bayes’ theorem 
#to compute probabilities.

nb.model = naiveBayes(HeartDisease~ Age + Sex + ChestPainType + RestingBP +
                        Cholesterol + FastingBS + RestingECG + MaxHR +
                        ExerciseAngina + Oldpeak + ST_Slope, data = train)

summary(nb.model)

# Predict probabilities for naive Bayes model
nb.prediction = predict(nb.model, newdata = test)
nb.prediction.probs = predict(nb.model, newdata = test, type = "raw")

predicted.classes = nb.prediction

conf_matrix_nb = table(predicted.classes, test$HeartDisease)
conf_matrix_nb

cm_nb <- confusionMatrix(predicted.classes, test$HeartDisease)
cm_nb$overall[c("Accuracy", "Kappa")]

#Accuracy : 0.8626374 
#Kappa : 0.7229327 


# ROC curve and AUC for Naive Bayes
roc_curve_naive = roc(test$HeartDisease, nb.prediction.probs[, 2])
auc(roc_curve_naive)

# Plot results
plot.roc(roc_curve_naive, col = "darkred", print.auc = TRUE, auc.polygon = TRUE,
         grid = c(0.1, 0.2), grid.col = c("darkblue", "darkblue"),
         max.auc.polygon = TRUE, auc.polygon.col = "lightcoral",
         print.thres = TRUE, legacy.axes = TRUE)



#The Naive Bayes model achieves an AUC of 0.93, indicating excellent predictive 
#ability. At the optimal probability threshold (≈ 0.58), it correctly identifies 
#88.9% of patients with heart disease and 86.1% of those without.Despite its 
#simplicity and strong independence assumption, Naive Bayes performs 
#nearly as well as logistic regression — showing it captures key patterns in 
#the predictors effectively.


#Conclusions
#Across all models, performance was strong but varied in complexity and stability. 
#Logistic regression achieved the highest discrimination (AUC ≈ 0.94), closely 
#followed by Naive Bayes (AUC ≈ 0.93), which worked surpisingly good despite
#its simplifying assumption of feature independence. The penalized logistic model 
#performed similarly while reducing overfitting through regularization. LDA (AUC ≈ 0.85) 
#and QDA (AUC ≈ 0.83) showed good results but were less accurate. Overall, 
#logistic and Naive Bayes models offered the best balance between predictive 
#power and interpretability.



#Decision tree
#A Decision Tree is a supervised machine learning algorithm used for classification and 
#regression tasks. It splits the data into subsets based on the most significant predictor 
#variables, forming a tree-like structure. Each node represents a decision rule, and each leaf 
#represents a class label or outcome. It is intuitive and easy to visualize, making it a widely 
#used algorithm for classification problems. 
tree <- rpart(HeartDisease~., data=data_clean, method = "class") 
rpart.plot(tree, digits=3) 



# Set the number of folds for cross-validation 
K=5  
# Create a factor vector to partition the training data into K folds. 
# `cut()` divides the row numbers of the dataset into K equally sized intervals. 
values=as.integer(cut(1:nrow(train), breaks=K)) 
folders=sample(values,size=nrow(train),replace = FALSE) 

#we are creating a grid of hyperparameter values to search through them. 
grid= expand.grid(maxdepth = 2:9,minsplit=seq(from=5,to=50, by=5), 
                  cp=2^(-3:-11),minbucket = seq(from = 5, to = 20, by = 5)) 

best_accuracy = 0 
best_params =c() 
#We are creating this for loop runs through each combination of hyperparameters that we 
#have defined in our grid before, giving me the best (biggest) accuracy.

for (i in 1:nrow(grid)) { 
  model = rpart(HeartDisease ~ ., data = train,  
                method = "class", 
                control=rpart.control(maxdepth = grid$maxdepth[i], 
                                      minsplit=grid$minsplit[i], 
                                      minbucket = grid$minbucket[i], 
                                      cp=grid$cp[i])) 
  test_prediction = predict(model, newdata = test, type = "class") 
  accuracy = mean(test_prediction == test$HeartDisease) 
  if (accuracy > best_accuracy) {
    best_accuracy = accuracy 
    best_params = grid[i,] 
  } 
}

#After the loop we can get the best hyperparameters that will give us the greatest accuracy 
accuracy  
print(best_params) 

#accuracy : 0.8516484
#maxdepth = 6, misnplit 5, cp = 0.0009765625,     minbucket = 5 

#Now we create our decision tree with the best hyperparameters 
best_model= rpart(HeartDisease ~ ., data = train,  
                  method = "class", 
                  control = rpart.control(maxdepth = 6, 
                                          minsplit = 5, 
                                          cp =0.0009765625, 
                                          minbucket = 5)) 

rpart.plot(best_model, digits=3) 
test_prediction2 = predict(best_model, newdata = test, type = "class") 


#We check that the accuracy of our decision tree is the same as the "best accuracy" calculated before 
accuracy2 = sum(test_prediction2 == test$HeartDisease) / nrow(test) 



#random forest
#Random Forest is an ensemble machine learning algorithm used for classification and 
#regression tasks. It works by constructing multiple decision trees and aggregating 
#their predictions. 
# Train initial random forest model 
randomf_model = randomForest(HeartDisease ~ .,  
                             data =train,  
                             importance = TRUE) 

print(randomf_model) 

#variable importance 
importance_df = data.frame( 
  Feature = row.names(importance(randomf_model)), 
  Importance = importance(randomf_model)[, 'MeanDecreaseGini'] 
) 

# Plot Feature Importances 
ggplot(importance_df,  
       aes(x = reorder(Feature, Importance),  
           y = Importance)) + 
  geom_bar(stat = 'identity', fill = 'skyblue3') +
  coord_flip() + 
  ggtitle('Feature Importances from Random Forest') + 
  xlab('') + 
  ylab('Mean Decrease in Gini') + 
  theme_minimal()

#Let's calculate the hyperparameters. We give different values for the hyperparameters to test. 
ntree_values = c(100, 500, 1000) 
mtry_values = c(1, 2, 3, 4, 5, 6, 7, 8, 9)  
nodesize_values = c(5, 10, 15) 
best_accuracy2 = 0 
best_hyperparameters2 = c() 
set.seed(123) 
#We create a loop like we did before in our decision tree to get the best hyperparameters 
#and accuracy for our random forest 
for (ntree in ntree_values) { 
  for (mtry in mtry_values) { 
    for (nodesize in nodesize_values) { 
      mrf_model = randomForest(HeartDisease ~ ., data = train, 
                               ntree = ntree, mtry = mtry, nodesize = nodesize) 
      mrf_prediction = predict(mrf_model, test) 
      mrf_accuracy = sum(mrf_prediction == test$HeartDisease) / nrow(test) 
      if (mrf_accuracy > best_accuracy2) { 
        best_accuracy2 = mrf_accuracy 
        best_hyperparameters2 = c(ntree = ntree, mtry = mtry, nodesize = nodesize) 
      } 
    } 
  } 
} 
#From the loop we get the best accuracy and the best hyperparameters for our model 
print(best_accuracy2)  
print(best_hyperparameters2) 
#for my model the best hyperparamters are ntree = 500,   mtry=4,   nodesize = 5 

#We store the accuracies 
mtry_values = c(1,2,3,4,5,6,7,8) 
accuracies = numeric(length(mtry_values)) 
for (i in seq_along(mtry_values)) { 
  set.seed(123)  
  #Train the random forest model with the current value of mtry 
  rf_model = randomForest(HeartDisease ~ ., data = train, 
                          ntree = 500, mtry = mtry_values[i], nodesize = 5) 
  #Create predicitions 
  rf_prediction = predict(rf_model, test)
  
  #Compute the accuracy 
  rf_accuracy = sum(rf_prediction == test$HeartDisease) / nrow(test) 
  #Store the accuracy 
  accuracies[i] = rf_accuracy 
} 

# Finally we plot our graph 
plot(mtry_values, accuracies, type = "b", pch = 20, 
     col = "darkcyan", xlab = "mtry", ylab = "Accuracy", 
     main = "Random Forest Accuracy vs. mtry") 

#We substitute these hyperparameters: RandomForest with Cross-Validation 
K_2 =5 
results_rf = numeric(K_2) 
for (i in 1:K_2) { 
  # Create our random forest model with cross-validation 
  rf_model = randomForest(HeartDisease ~ ., data = train, 
                          ntree = 100, mtry = 8, nodesize = 15) 
  # Make aprediction in our testing data 
  rf_prediction = predict(rf_model, test) 
  # Get the accuracy of our random forest 
  rf_accuracy = sum(rf_prediction == test$HeartDisease) / nrow(test) 
  print(rf_accuracy) 
  # Store results 
  results_rf[i] = rf_accuracy 
} 

#We will graph the error for a better understanding of the random forest. 
classifier = randomForest(formula = HeartDisease~., 
                          data = data_clean) 
plot(classifier) 

#Now we obtain the importance of the variables of our random forest: 
variable_importance = importance(rf_model) 
# Print the variable importance 
print(variable_importance) 

#The Mean Decrease Gini indicates how much each variable contributes to improving the 
#model's accuracy, based on the Gini impurity criterion.  



#KNN 
#KNN classifies each observation based on the majority class of its k nearest 
#neighbors in the feature space.
# Select only numeric predictor variables (KNN uses distances)
num_training <- train[, c("RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "Age")]
num_testing  <- test[,  c("RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "Age")]

# Extract the outcome variable (class labels)
train_l <- train$HeartDisease
test_l  <- test$HeartDisease

# Create a vector to store accuracy values for each k
accuracy_values <- numeric(20)

# Try k values from 1 to 20
for (k in 1:20) {
  # Fit KNN with k neighbors
  knn_model <- class::knn(train = num_training,
                          test  = num_testing,
                          cl    = train_l,
                          k     = k)
  
  # Compute accuracy on the test set
  accuracy_values[k] <- mean(knn_model == test_l)
}

# Find the k value with the highest accuracy
best_k <- which.max(accuracy_values)
best_accuracy <- max(accuracy_values)

best_k
best_accuracy

# Add the outcome variable back for caret
num_training$HeartDisease <- train_l

# Set up 10-fold cross-validation
train_control <- trainControl(method = "cv", number = 10)

# Train KNN using cross-validation
knn_cv_model <- train(HeartDisease ~ .,
                      data = num_training,
                      method = "knn",
                      trControl = train_control)

# Display cross-validation results and best k
print(knn_cv_model)


#The manually tuned KNN achieved an accuracy of about 72% on the test set, with 
#k = 4 selected as the best value. Using 10-fold cross-validation, the KNN model
#selected k = 9 and achieved a slightly lower accuracy of around 70%. The manually 
#tuned KNN achieved a slightly higher accuracy on the test set; however, the 
#cross-validated KNN provides a more reliable estimate of performance, as it 
#averages results across multiple training and validation splits. Overall, KNN 
#shows moderate predictive performance and performs worse than logistic regression 
#models considered earlier.



















# ================================================================
# RECONSTRUCTED SECTION 4 — FEATURE SELECTION
# Recreated from PROJECT2-REPORT.pdf; original section unavailable.
# Uses the existing train/test split and data_clean created above.
# ================================================================

# The report describes 3 logistic-regression models. Its "15 predictors"
# count does not match the 11 original predictors + 3 engineered flags
# present in this script (14 input columns before dummy encoding).

full_predictors <- c(
  "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
  "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
  "Oldpeak", "ST_Slope", "HighRisk", "HighRiskAngina", "HighRiskST"
)

# The six features identified as important by the Random Forest in the report.
# They are fixed here to reproduce the reported comparison; they are NOT
# reselected using test-set performance.
rf_top6_predictors <- c(
  "ST_Slope", "ChestPainType", "MaxHR", "Oldpeak", "Cholesterol", "Age"
)

risk_predictors <- c(
  "ChestPainType", "ST_Slope", "ExerciseAngina", "Oldpeak",
  "HighRiskST", "HighRiskAngina"
)

# Evaluate every model on the same untouched test set.
# ROC AUC is calculated for the "Yes" class.
run_feature_model <- function(model_name, predictors, train_data, test_data) {
  model_formula <- reformulate(predictors, response = "HeartDisease")
  fit <- glm(model_formula, family = binomial(), data = train_data)
  probabilities <- predict(fit, newdata = test_data, type = "response")
  predictions <- factor(
    ifelse(probabilities >= 0.5, "Yes", "No"),
    levels = levels(train_data$HeartDisease)
  )
  truth <- test_data$HeartDisease
  cm <- caret::confusionMatrix(predictions, truth, positive = "Yes")
  auc_value <- as.numeric(pROC::auc(
    pROC::roc(truth, probabilities, levels = c("No", "Yes"),
              direction = "<", quiet = TRUE)
  ))
  list(
    fit = fit,
    results = data.frame(
      Model = model_name,
      Accuracy = unname(cm$overall["Accuracy"]),
      AUC = auc_value,
      Kappa = unname(cm$overall["Kappa"]),
      row.names = NULL
    )
  )
}

full_result <- run_feature_model("Full model", full_predictors, train, test)
top6_result <- run_feature_model("Top 6 RF", rf_top6_predictors, train, test)
risk_result <- run_feature_model("Risk-focused", risk_predictors, train, test)

feature_comparison <- rbind(
  full_result$results,
  top6_result$results,
  risk_result$results
)
print(feature_comparison)

# Values transcribed from the submitted report for comparison only.
# Do not treat these as newly reproduced results.
reported_results <- data.frame(
  Model = c("Full model", "Top 6 RF", "Risk-focused"),
  Report_Accuracy = c(0.879, 0.868, 0.846),
  Report_AUC = c(0.933, 0.902, 0.907),
  Report_Kappa = c(0.755, 0.735, 0.691)
)
print(merge(feature_comparison, reported_results, by = "Model", sort = FALSE))

# IMPORTANT REPRODUCIBILITY NOTES:
# - This reconstructs the described models, not the missing original code.
# - Results may differ from the PDF because the exact original workflow is unknown.
# - The earlier script imputes cholesterol before splitting the dataset,
#   introducing potential data leakage. A publication-quality refactor should
#   fit preprocessing on training data only.
# - Earlier KNN tuning uses the test set to select k. This should be corrected
#   before using those KNN results as an unbiased comparison.
