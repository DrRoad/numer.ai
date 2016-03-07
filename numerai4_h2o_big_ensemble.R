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
h2o.glm.1 <-
  function(..., alpha = 0.0, solver = 'L_BFGS')
    h2o.glm.wrapper(..., alpha = alpha, solver = solver)
h2o.glm.2 <-
  function(..., alpha = 0.5, solver = 'L_BFGS')
    h2o.glm.wrapper(..., alpha = alpha, solver = solver)
h2o.glm.3 <-
  function(..., alpha = 1.0, solver = 'L_BFGS')
    h2o.glm.wrapper(..., alpha = alpha, solver = solver)
h2o.randomForest.1 <-
  function(..., ntrees = 200, nbins = 50, seed = 1234)
    h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.randomForest.2 <-
  function(..., ntrees = 200, sample_rate = 0.75, seed = 1234)
    h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.3 <-
  function(..., ntrees = 200, sample_rate = 0.85, seed = 1234)
    h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.gbm.1 <-
  function(..., ntrees = 100, seed = 1234)
    h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)
h2o.gbm.2 <-
  function(..., ntrees = 100, nbins = 50, seed = 1234)
    h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.gbm.3 <-
  function(..., ntrees = 100, max_depth = 10, seed = 1234)
    h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.deeplearning.1 <-
  function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, seed = 1234)
    h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.2 <-
  function(..., hidden = c(200,200,200), activation = "Tanh", epochs = 50, seed = 1234)
    h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.3 <-
  function(..., hidden = c(500,500), activation = "RectifierWithDropout", epochs = 50, seed = 1234)
    h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)

learner <- c("h2o.glm.1", "h2o.glm.2", "h2o.glm.3",
             "h2o.randomForest.1", "h2o.randomForest.2", "h2o.randomForest.3",
             "h2o.gbm.1", "h2o.gbm.2", "h2o.gbm.3", 
             "h2o.deeplearning.1", "h2o.deeplearning.2", "h2o.deeplearning.3") 
metalearner <- "h2o.glm.wrapper"

# ------------------------------------------------------------------------------------------------------------------
# Train an ensemble of 12 base learners and one metalearner and create tournament predictions -------------------
model_ens_big <- h2o.ensemble(x = feature_names, 
                               y = "target", 
                               training_frame = train_h2o, 
                               family = "binomial", 
                               learner = learner, 
                               metalearner = metalearner,
                               cvControl = list(V = 10))


pred_ens_big <- predict(model_ens_big, tourn_h2o)
pred <- as.data.frame(pred_ens_big$pred)[,3]  #third column is P(Y==1)
prob <- cbind(tourn[, c("t_id"), drop = FALSE], probability = pred)
write.table(prob[, c("t_id", "probability")], paste0("h2o_big_ens_CV10", ".csv"), sep = ",", row.names = FALSE)
# LOGLOSS @ NUMERAI - 0.6916
# The result is worse than in the case of simple ensemble but shows a few bucks for originality meaning that it had low correlation with other results


# Shutdown H2O --------------------
h2o.shutdown()
