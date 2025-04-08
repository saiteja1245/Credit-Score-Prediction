# Load necessary libraries
library(sqldf)
library(class)
library(gmodels)
library(rpart)
library(rpart.plot)
library(caret)
library(ggplot2)
library(e1071)
# Load the dataset
dataset <- read.csv(file.choose())
View(dataset)
# View the structure and check for missing values
str(dataset)
colSums(is.na(dataset))

# Select relevant columns for analysis
dataset <- dataset[, c("Age", "Monthly_Inhand_Salary", "Total_EMI_per_month", 
                       "Amount_invested_monthly", "Payment_Behaviour", 
                       "Changed_Credit_Limit", "Credit_Score")]

# Convert columns to numeric, where applicable
dataset$Age <- as.numeric(dataset$Age)
dataset$Amount_invested_monthly <- as.numeric(dataset$Amount_invested_monthly)
dataset$Changed_Credit_Limit <- as.numeric(dataset$Changed_Credit_Limit)

# Check distinct values in categorical columns
sqldf("select DISTINCT Payment_Behaviour from dataset")
sqldf("select DISTINCT Credit_Score from dataset")

# Handle missing values
colSums(is.na(dataset))
dataset$Age[is.na(dataset$Age)] <- sample(100, sum(is.na(dataset$Age)), replace = TRUE)
dataset <- dataset[dataset$Age > 0 & dataset$Age < 200, ]
dataset$Monthly_Inhand_Salary[is.na(dataset$Monthly_Inhand_Salary)] <- mean(dataset$Monthly_Inhand_Salary, na.rm = TRUE)
dataset$Amount_invested_monthly[is.na(dataset$Amount_invested_monthly)] <- mean(dataset$Amount_invested_monthly, na.rm = TRUE)
dataset$Changed_Credit_Limit[is.na(dataset$Changed_Credit_Limit)] <- mean(dataset$Changed_Credit_Limit, na.rm = TRUE)

# Handle special cases in 'Total_EMI_per_month' column
dataset$Total_EMI_per_month[dataset$Total_EMI_per_month == "-"] <- NA
dataset$Total_EMI_per_month <- as.numeric(dataset$Total_EMI_per_month)
dataset$Total_EMI_per_month[is.na(dataset$Total_EMI_per_month)] <- mean(dataset$Total_EMI_per_month, na.rm = TRUE)

# Split the dataset into training and test sets
set.seed(123)
samplee <- sample(2, nrow(dataset), replace = TRUE, prob = c(0.70, 0.30))
train_set <- dataset[samplee == 1, ]
test_set <- dataset[samplee == 2, ]

# Normalize selected numerical columns
normalize <- function(x) { (x - min(x)) / (max(x) - min(x)) }
dataset_norm <- as.data.frame(lapply(dataset[, c("Age", "Monthly_Inhand_Salary", "Total_EMI_per_month")], normalize))

train_set_norm <- dataset_norm[samplee == 1, ]
test_set_norm <- dataset_norm[samplee == 2, ]
train_label <- dataset[samplee == 1, "Credit_Score"]
test_label <- dataset[samplee == 2, "Credit_Score"]

# Define accuracy function
accuracy <- function(x) { sum(diag(x) / sum(x)) * 100 }

# K-Nearest Neighbors
knnn <- knn(train_set_norm, test_set_norm, cl = train_label, k = 3)
CrossTable(x = test_label, y = knnn, prop.chisq = FALSE)
tab_knn <- table(knnn, test_label)
knn_accuracy <- accuracy(tab_knn)
cat("KNN Accuracy:", knn_accuracy, "%\n")

# Naive Bayes
train_set$Credit_Score <- factor(train_set$Credit_Score)
test_set$Credit_Score <- factor(test_set$Credit_Score, levels = levels(train_set$Credit_Score))
nb_model <- naiveBayes(Credit_Score ~ ., data = train_set)
nb_predictions <- predict(nb_model, test_set)
nb_conf_matrix <- confusionMatrix(nb_predictions, test_set$Credit_Score)
CrossTable(x = test_set$Credit_Score, y = nb_predictions, prop.chisq = FALSE)
nb_accuracy <- accuracy(nb_conf_matrix$table)
cat("Naive Bayes Accuracy:", nb_accuracy, "%\n")

# Decision Tree
train_set$Credit_Score <- factor(train_set$Credit_Score)
test_set$Credit_Score <- factor(test_set$Credit_Score, levels = levels(train_set$Credit_Score))
tree <- rpart(Credit_Score ~ ., data = train_set, method = "class")
rpart.plot(tree, box.palette = "RdYlGn")
tree_pred <- predict(tree, test_set, type = "class")
conf_matrix_tree <- confusionMatrix(tree_pred, test_set$Credit_Score)
tree_accuracy <- accuracy(conf_matrix_tree$table)
cat("Decision Tree Accuracy:", tree_accuracy, "%\n")

#Random Forest
library(randomForest)
rf_model <- randomForest(Credit_Score!., data = train_set)
rf_predictions <- predict(rf_model, test_set)

#Confusion Matrix and Accuracy for Random Forest
rf_conf_matrix <- confusionMatrix(rf_predictions, test_set$Credit_Score)
rf_accuracy <- accuracy(rf_conf_matrix$table)
cat("Random Forest Accuracy:", rf_accuracy,"%\n")
importance_rf <- importance(rf_model)
importance_rf
varImpPlot(rf_model,
           main = "Random Forest - Feature Importance",
           col = c("blue","darkgreen"),
           pch = 16,
           cex = 1.5)


# Display all accuracies
cat("KNN Accuracy:", knn_accuracy, "%\n")
cat("Naive Bayes Accuracy:", nb_accuracy, "%\n")
cat("Decision Tree Accuracy:", tree_accuracy, "%\n")
