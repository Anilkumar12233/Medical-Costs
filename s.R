library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(reshape2)
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(gbm)
library(xgboost)

df <- read.csv("~/STA 631/Activities/Medical-Costs/insurance.csv")
df(head)
# Check for null values
sum(is.na(df))



# Encoding categorical variables
df$sex <- as.numeric(factor(df$sex))
df$smoker <- as.numeric(factor(df$smoker))
df$region <- as.numeric(factor(df$region))

# Calculating correlation matrix
corr <- round(cor(df), 2)

# Creating heatmap
plt <- ggplot(melt(corr), aes(Var2, Var1, fill = value)) +
  geom_tile() +
  geom_text(aes(label = value), color = "black", size = 3) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(x = "", y = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
        axis.text.y = element_text(hjust = 0.5, vjust = 0.5),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())

# Displaying the heatmap
print(plt)

library(ggplot2)

# Distribution plot for smokers
p1 <- ggplot(subset(df, smoker == 1), aes(x = charges)) +
  geom_density(fill = "cyan", color = "black") +
  labs(title = "Distribution of charges for smokers")

print(p1)

# Create the count plot
ggplot(df, aes(x = smoker, fill = sex)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("pink", "white")) +
  labs(x = "Smoker", y = "Count") +
  ggtitle("Count of Smokers by Gender") +
  theme_bw()



# Subset the data for age == 18
subset_data <- subset(df, age == 18)

# Create the count plot
ggplot(subset_data, aes(x = smoker, fill = sex)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = rainbow(length(unique(subset_data$sex)))) +
  labs(x = "Smoker", y = "Count") +
  ggtitle("The number of smokers and non-smokers (18 years old)") +
  theme_bw()

ggplot(df, aes(x = smoker, y = charges, fill = smoker)) +
  geom_bar(stat = "identity", position = "identity") +
  labs(x = "Smoker", y = "Charges") +
  ggtitle("Distribution of Charges by Smoker") +
  theme_bw()

# Convert smoker to factor
df$smoker <- as.factor(df$smoker)

# Create the scatter plot with linear regression line
ggplot(df, aes(x = age, y = charges, color = smoker)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "Age", y = "Charges") +
  ggtitle("Relationship between Age and Charges") +
  theme_bw()

ggplot(df, aes(x = bmi)) +
  geom_histogram(binwidth = 1, fill = "lightblue", color = "black") +
  geom_density(fill = "lightgreen", alpha = 0.5) +
  labs(x = "BMI", y = "Frequency") +
  ggtitle("Distribution of BMI") +
  theme_bw()

ggplot(df[df$bmi >= 30, ], aes(x = charges)) +
  geom_density(fill = "lightblue", color = "black") +
  labs(x = "Charges", y = "Density") +
  ggtitle("Distribution of Charges for BMI > 30") +
  theme_bw()
ggplot(df[df$bmi < 30, ], aes(x = charges)) +
  geom_density(fill = "lightblue", color = "black") +
  labs(x = "Charges", y = "Density") +
  ggtitle("Distribution of Charges for BMI < 30") +
  theme_bw()


# Scatter plot
ggplot(df, aes(x = bmi, y = charges, color = smoker)) +
  geom_point() +
  labs(x = "BMI", y = "Charges") +
  ggtitle("Scatter Plot of Charges and BMI") +
  theme_bw()

X <- df[, !(names(df) %in% c("charges"))]
y <- df$charges

library(caret)

# Set the random seed for reproducibility
set.seed(123)

# Split the data into training and testing sets
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]


# Fit the scaler on the training set
scaler <- preProcess(X_train, method = "center", "scale")

# Transform the training set
X_train <- predict(scaler, X_train)

# Transform the testing set
X_test <- predict(scaler, X_test)

library(class)

regressors <- list(
  'Linear Regression' = lm,
  'Decision Tree' = rpart,
  'Random Forest' = randomForest
)

results <- data.frame(matrix(ncol = 3, nrow = length(regressors)))
colnames(results) <- c('MAE', 'MSE', 'R2-score')

for (method in names(regressors)) {
  if (method == 'Linear Regression') {
    model <- regressors[[method]](formula = y_train ~ ., data = cbind(X_train, y_train))
  } else {
    model <- regressors[[method]](formula = y_train ~ ., data = X_train)
  }
  
  pred <- predict(model, newdata = X_test)
  
  results[method, 'MAE'] <- round(mean(abs(pred - y_test)), 3)
  results[method, 'MSE'] <- round(mean((pred - y_test)^2), 3)
  results[method, 'R2-score'] <- round(cor(pred, y_test)^2, 3)
}

for (method in names(regressors)) {
  if (method == 'Linear Regression') {
    model <- regressors[[method]](formula = y_train ~ ., data = cbind(X_train, y_train))
  } else {
    model <- regressors[[method]](formula = y_train ~ ., data = X_train)
  }
  
  pred <- predict(model, newdata = X_test)
  
  if (anyNA(pred)) {
    results[method, 'MAE'] <- NA
    results[method, 'MSE'] <- NA
    results[method, 'R2-score'] <- NA
  } else {
    results[method, 'MAE'] <- round(mean(abs(pred - y_test)), 3)
    results[method, 'MSE'] <- round(mean((pred - y_test)^2), 3)
    results[method, 'R2-score'] <- round(cor(pred, y_test)^2, 3)
  }
}

results <- results[order(-results$`R2-score`), ]
print(results)





