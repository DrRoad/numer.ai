library(xgboost)

# Get training and tournament data -------------------
train <- read.csv("http://datasets.numer.ai/numerai_training_data.csv")
tourn <- read.csv("http://datasets.numer.ai/numerai_tournament_data.csv")

# Get feature names -------------------
feature_names <- names(train)[1:(ncol(train)-1)]

# Convert data to XGBost matrix --------------------
train_xgb = xgb.DMatrix(data.matrix(train[, feature_names]), label = train[, c("target")])
tourn_xgb <- xgb.DMatrix(data.matrix(tourn))

# Train XGBost model and create tournament predictions -------------------
set.seed(1234) # set seed for reproducability
model_xgb = xgboost(data = train_xgb, nrounds = 100, eval_metric = "logloss", objective = "binary:logistic", verbose = 2)
# output
# tree prunning end, 1 roots, 126 extra nodes, 0 pruned nodes ,max_depth=6
# [0]	train-logloss:0.689731
# ...
# tree prunning end, 1 roots, 98 extra nodes, 0 pruned nodes ,max_depth=6
# [99]	train-logloss:0.574348
# Results on training data only seem very encouraging considering that the leader's LOGLOSS is 0.6897

pred_xgb <- predict(model_xgb, newdata = tourn_xgb)
prob <- cbind(tourn[, c("t_id"), drop = FALSE], data.frame(xgb = pred_xgb))
write.table(prob[, c("t_id", "xgb")], "xgb_booster.csv", sep = ",", row.names = FALSE, col.names = c("t_id", "probability"))

# Upload to NUMERAI results in LOGLOSS of 0.7657 which would place one at the very bottom of the leaderboard :-(
# TODO: Hyperparameter tuning and feature engineering
