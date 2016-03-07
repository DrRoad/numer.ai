library(h2o)

# Initiate H2O --------------------
h2o.removeAll() # Clean up. Just in case H2O was already running
h2o.init(nthreads = -1, max_mem_size="22G")  # Start an H2O cluster with all threads available

# Get training and tournament data -------------------
train <- read.csv("http://datasets.numer.ai/numerai_training_data.csv")
tourn <- read.csv("http://datasets.numer.ai/numerai_tournament_data.csv")

# Convert target to factor -------------------
train$target <- as.factor(train$target)

# Get feature names -------------------
feature_names <- names(train)[1:(ncol(train)-1)]

# Convert to H20 objects -------------------
train_h2o <- as.h2o(train)
tourn_h2o <- as.h2o(tourn)

# Create the data frame with predicted probabilities ------------------
prob <- tourn[, "t_id", drop = FALSE]

# ------------------------------------------------------------------------------------------------------------------
# Train a GLM model and create tournament predictions -------------------
# GLM doesn't use logloss
model_glm <- h2o.glm(x = feature_names,  y = "target", training_frame = train_h2o)
h2o.performance(model_glm) 
# output ----------
# MSE:  0.2490025
# R2 :  0.00377694
# Mean Residual Deviance :  0.2490025
# Null Deviance :14439.66
# Null D.o.F. :57770
# Residual Deviance :14385.12
# Residual D.o.F. :57749
# AIC :83674.41

pred_glm <- predict(model_glm, newdata = tourn_h2o)
prob <- cbind(prob, as.data.frame(pred_glm$predict, col.names = "glm"))
write.table(prob[, c("t_id", "glm")], paste0(model_glm@model_id, ".csv"), sep = ",", row.names = FALSE, col.names = c("t_id", "probability"))
# LOGLOSS @ NUMERAI - 0.6910


# ------------------------------------------------------------------------------------------------------------------
# Train a GBM model and create tournament predictions -------------------
model_gbm <- h2o.gbm(x = feature_names,  y = "target", training_frame = train_h2o, stopping_metric = "logloss")
h2o.logloss(model_gbm)
# output ----------
# [1] 0.6759121

pred_gbm <- predict(model_gbm, newdata = tourn_h2o)
prob <- cbind(prob, as.data.frame(pred_gbm$p1, col.names = "gbm"))
write.table(prob[, c("t_id", "gbm")], paste0(model_gbm@model_id, ".csv"), sep = ",", row.names = FALSE, col.names = c("t_id", "probability"))
# LOGLOSS @ NUMERAI - 0.6928


# ------------------------------------------------------------------------------------------------------------------
# Train a random forest model and create tournament predictions -------------------
model_rf <- h2o.randomForest(x = feature_names, y = "target", training_frame = train_h2o, stopping_metric = "logloss")
h2o.logloss(model_rf)
# output ----------
# [1] 0.7153093

pred_rf <- predict(model_rf, newdata = tourn_h2o)
prob <- cbind(prob, as.data.frame(pred_rf$p1, col.names = "rf"))
write.table(prob[, c("t_id", "rf")], paste0(model_rf@model_id, ".csv"), sep = ",", row.names = FALSE, col.names = c("t_id", "probability"))
# LOGLOSS @ NUMERAI - 0.7002


# ------------------------------------------------------------------------------------------------------------------
# Train a deep learning model and create tournament predictions -------------------
model_dl <- h2o.deeplearning(x = feature_names, y = "target", training_frame = train_h2o, stopping_metric = "logloss")
h2o.logloss(model_dl)
# output ----------
# 0.6878023

pred_dl <- predict(model_dl, newdata = tourn_h2o)
prob <- cbind(prob, as.data.frame(pred_dl$p1, col.names = "dl"))
write.table(prob[, c("t_id", "dl")], paste0(model_dl@model_id, ".csv"), sep = ",", row.names = FALSE, col.names = c("t_id", "probability"))
# LOGLOSS @ NUMERAI - 0.7031

# -----------------------------------------------------------------------------------------------------
# Check how predictions between 4 models correlate
cor(prob[, -1])
# output ----------
#           glm       gbm        rf        dl
# glm 1.0000000 0.6386357 0.3237184 0.4099687
# gbm 0.6386357 1.0000000 0.4093481 0.4187918
# rf  0.3237184 0.4093481 1.0000000 0.2681444
# dl  0.4099687 0.4187918 0.2681444 1.0000000

# Since the predictions are close and not very correlated see how averaging performs @ NUMERAI
prob$mean <- rowMeans(prob[, -1])
write.table(prob[, c("t_id", "mean")], paste0("H2O_glm_gbm_rf_dl", ".csv"), sep = ",", row.names = FALSE, col.names = c("t_id", "probability"))
# LOGLOSS @ NUMERAI - 0.6932
# Simple averaging didn't improve the best result produced by GLM model


# Shutdown H2O --------------------
h2o.shutdown()

