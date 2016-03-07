library(h2o)
library(h2oEnsemble)

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

# Setup learners --------------------
learner <- c("h2o.glm.wrapper", "h2o.randomForest.wrapper", "h2o.gbm.wrapper", "h2o.deeplearning.wrapper") #
metalearner <- "h2o.glm.wrapper"

# ------------------------------------------------------------------------------------------------------------------
# Train a simple ensemble and create tournament predictions -------------------
model_ens_simp <- h2o.ensemble(x = feature_names, 
                               y = "target", 
                               training_frame = train_h2o, 
                               family = "binomial", 
                               learner = learner, 
                               metalearner = metalearner,
                               cvControl = list(V = 10))


pred_ens_simp <- predict(model_ens_simp, tourn_h2o)
pred <- as.data.frame(pred_ens_simp$pred)[,3]  #third column is P(Y==1)
prob <- cbind(tourn[, c("t_id"), drop = FALSE], probability = pred)
write.table(prob[, c("t_id", "probability")], paste0("h2o_simp_ens_CV10", ".csv"), sep = ",", row.names = FALSE)
# LOGLOSS @ NUMERAI - 0.6912

# Shutdown H2O --------------------
h2o.shutdown()