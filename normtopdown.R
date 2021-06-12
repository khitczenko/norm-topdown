####################################################################################################################################
library(dplyr)
library("caret")
library("glmnet")

GLOBAL_USE_ADS <- TRUE
GLOBAL_USE_POS <- TRUE
GLOBAL_USE_WEIGHTS <- FALSE

# Read in data from a particular directory
data <- prep_data('/Users/khit/OverlappingCats/', usePOS=GLOBAL_USE_POS, useADS=GLOBAL_USE_ADS)

##############################################################################
##############################################################################

# THEORY 1: Normalization/Factoring out systematic variability (linear regression)
avg_labels <- c('regressed all', 'regressed short', 'regressed long', 'baseline all', 'baseline short', 'baseline long')
avgs <- c(0,0,0,0,0,0)

for(ITERATIONS in 1:10) {
  # Shuffle the data
  shuffled <- shuffle_data(data)
  train <- shuffled$train
  test <- shuffled$test
  
  ##############################################
  ##########        MULTITARGET      ###########
  ##############################################
  # To be predicted
  targets <- c('duration', 'f1', 'f2', 'f3')
  # contextual factors
  inputs <- 'pos + quality + speaker + accented + wi + wm + wf + ai + am + af + ii + im + iif + ui + um + uf + bi + bi_start + prev + foll + utt_rate + word_rate + prevduration + follduration'
  
  # Run linear regression
  linreg <- do_multitarget_linear_regression(targets, inputs, train, test)
  train <- linreg$train
  test <- linreg$test
  # Run logistic regression to predict short/long
  logreg <- train_and_predict_glm(paste0('length ~ ', paste0('residuals_', targets, collapse=' + ')), train, test, use_weights=GLOBAL_USE_WEIGHTS)
  # Keep track of overall accuracy as well as accuracy on short/long vowels
  avgs[1] <- avgs[1] + logreg$accuracy
  avgs[2] <- avgs[2] + logreg$short.accuracy
  avgs[3] <- avgs[3] + logreg$long.accuracy
  # Print results
  print(paste0('Regression model BIC: ', logreg$BIC, ', Accuracy: ', logreg$accuracy, ', Short Accuracy: ', logreg$short.accuracy, ', Long Accuracy: ', logreg$long.accuracy))

  # Comparison model where you just use duration to predict length without regressing anything out
  # Train model, then make predictions, and get accuracy, etc.
  baseline_logreg <- train_and_predict_glm(paste0('length ~ ', paste0(targets, collapse=' + ')), train, test, use_weights=GLOBAL_USE_WEIGHTS)
  baseline_model <- baseline_logreg$model
  avgs[4] <- avgs[4] + baseline_logreg$accuracy
  avgs[5] <- avgs[5] + baseline_logreg$short.accuracy
  avgs[6] <- avgs[6] + baseline_logreg$long.accuracy
  print(paste0('Baseline model BIC: ', baseline_logreg$BIC, ', Accuracy: ', baseline_logreg$accuracy, ', Short Accuracy: ', baseline_logreg$short.accuracy, ', Long Accuracy: ', baseline_logreg$long.accuracy))
}
# Print overall average (divide by 10 as we're running 10 iterations)
for(i in 1:6) {
  print(paste0(avg_labels[i], ': ', avgs[i]/10))
}

##################################################################################################################################################################
##################################################################################################################################################################

# THEORY 2: TOP-DOWN INFORMATION
avg_labels <- c('topdown all', 'topdown short', 'topdown long', 'baseline all', 'baseline short', 'baseline long')
avgs <- c(0,0,0,0,0,0)

for(ITERATIONS in 1:10) {
  shuffled <- shuffle_data(data)
  train <- shuffled$train
  test <- shuffled$test

  ##############################################
  ########## PREDICTION FORMULA ################
  ##############################################
  form <- 'duration + quality + speaker + accented + wi + wm + wf + ai + am + af + ii + im + iif + ui + um + uf + bi + bi_start + prev + foll + utt_rate + word_rate + prevduration + follduration + prev_gem + foll_gem + f1 + f2 + f3'
  if (GLOBAL_USE_ADS == FALSE) {
    form <- paste0('condition + ', form)
  }
  if(GLOBAL_USE_POS) {
    form <- paste0('pos + ', form)
  }
  form <- paste0('length ~ ', form)
  
  # Run logistic regression using all contextual factors
  logreg <- train_and_predict_glm(form, train, test, use_weights=GLOBAL_USE_WEIGHTS)
  avgs[1] <- avgs[1] + logreg$accuracy
  avgs[2] <- avgs[2] + logreg$short.accuracy
  avgs[3] <- avgs[3] + logreg$long.accuracy
  print(paste0('Regression model BIC: ', logreg$BIC, ', Accuracy: ', logreg$accuracy, ', Short Accuracy: ', logreg$short.accuracy, ', Long Accuracy: ', logreg$long.accuracy))  
  
  # Comparison model where you just use duration to predict length without regressing anything out
  # Train model, then make predictions, and get accuracy, etc.
  baseline_logreg <- train_and_predict_glm('length ~ duration', train, test, use_weights=GLOBAL_USE_WEIGHTS)
  baseline_model <- baseline_logreg$model
  avgs[4] <- avgs[4] + baseline_logreg$accuracy
  avgs[5] <- avgs[5] + baseline_logreg$short.accuracy
  avgs[6] <- avgs[6] + baseline_logreg$long.accuracy
  print(paste0('Baseline model BIC: ', baseline_logreg$BIC, ', Accuracy: ', baseline_logreg$accuracy, ', Short Accuracy: ', baseline_logreg$short.accuracy, ', Long Accuracy: ', baseline_logreg$long.accuracy))
}
for(i in 1:6) {
  print(paste0(avg_labels[i], ': ', avgs[i]/10))
}

###############################################################################################
## Functions 
###############################################################################################

# Get data ready for running
prep_data <- function(working_dir,usePOS=FALSE, useADS=FALSE) {
  setwd(working_dir)
  data <- read.table('vowel_features_v2_withtest.tsv', header = TRUE, fill = TRUE)
  
  # Factorize anything that is a discrete variable
  data$filename <- factor(data$filename)
  data$segment <- factor(data$segment)
  data$quality <- factor(data$quality)
  data$length <- factor(data$length)
  data$speaker <- factor(data$speaker)
  data$condition <- factor(data$condition)
  data$accented <- factor(data$accented)
  data$wi <- factor(data$wi)
  data$wm <- factor(data$wm)
  data$wf <- factor(data$wf)
  data$ai <- factor(data$ai)
  data$am <- factor(data$am)
  data$af <- factor(data$af)
  data$ii <- factor(data$ii)
  data$im <- factor(data$im)
  data$iif <- factor(data$iif)
  data$ui <- factor(data$ui)
  data$um <- factor(data$um)
  data$uf <- factor(data$uf)
  data$pos <- factor(data$pos)
  data$bi <- factor(data$bi)
  data$prev <- factor(data$prev)
  data$prev_gem <- factor(data$prev_gem)
  data$foll <- factor(data$foll)
  data$foll_gem <- factor(data$foll_gem)
  data$is_first_word_of_APIPUtt <- factor(data$is_first_word_of_APIPUtt)
  data$istest <- factor(data$istest)
  
  # Split data into ADS and IDS
  adsdata <- subset(data, condition == 'A')
  idsdata <- subset(data, condition != 'A')
  
  # Pick ADS or IDS
  if(useADS) {
    data <- adsdata
  } else {
    data <- idsdata
  }
  
  f <- c('AUX', 'AUX_PRT', 'CONJ', 'PRON', 'PRON_PRT', 'PRT', 'PRT_PRT', 'SFX', 'SFX_PRT', 'INTJ', 'PRF')
  c <- c('ADJ', 'ADN', 'ADN_NOUN', 'ADJ_ADJ', 'ADV', 'ADV_NOUN', 'ADV_VERB', 'NOUN', 'NOUN_VERB', 'VERB', 'ADT', 'SYM')
  
  data$pos <- 1*(data$pos %in% f) + 2*(data$pos %in% c)
  data$pos <- factor(data$pos)
  
  # Get rid of 'A' stuff
  data$condition <- factor(data$condition)
  data$filename <- factor(data$filename)
  
  # Shuffle data
  datashuffled <- data[sample(nrow(data)),]
  
  if(usePOS) {
    # Get the subset of columns that I want - INCLUDE POS
    data <- data.frame(datashuffled$length, datashuffled$quality, datashuffled$duration, datashuffled$speaker, datashuffled$condition, datashuffled$accented, datashuffled$wi, datashuffled$wm, datashuffled$wf, datashuffled$ai, datashuffled$am, datashuffled$af, datashuffled$ii, datashuffled$im, datashuffled$iif, datashuffled$ui, datashuffled$um, datashuffled$uf, datashuffled$pos, datashuffled$bi, datashuffled$is_first_word_of_APIPUtt, datashuffled$prev, datashuffled$prev_gem, datashuffled$foll, datashuffled$foll_gem, datashuffled$f1, datashuffled$f2, datashuffled$f3, datashuffled$utt_rate, datashuffled$word_rate, datashuffled$prevduration, datashuffled$follduration, datashuffled$istest)
    colnames(data) <- c('length', 'quality', 'duration', 'speaker', 'condition', 'accented', 'wi', 'wm', 'wf', 'ai', 'am', 'af', 'ii', 'im', 'iif', 'ui', 'um', 'uf', 'pos', 'bi', 'bi_start', 'prev', 'prev_gem', 'foll', 'foll_gem', 'f1', 'f2', 'f3', 'utt_rate', 'word_rate', 'prevduration', 'follduration', 'istest')
  } else { 
    # Get the subset of columns that I want - EXCLUDE POS
    data <- data.frame(datashuffled$length, datashuffled$quality, datashuffled$duration, datashuffled$speaker, datashuffled$condition, datashuffled$accented, datashuffled$wi, datashuffled$wm, datashuffled$wf, datashuffled$ai, datashuffled$am, datashuffled$af, datashuffled$ii, datashuffled$im, datashuffled$iif, datashuffled$ui, datashuffled$um, datashuffled$uf, datashuffled$bi, datashuffled$is_first_word_of_APIPUtt, datashuffled$prev, datashuffled$prev_gem, datashuffled$foll, datashuffled$foll_gem, datashuffled$f1, datashuffled$f2, datashuffled$f3, datashuffled$utt_rate, datashuffled$word_rate, datashuffled$prevduration, datashuffled$follduration, datashuffled$istest)
    colnames(data) <- c('length', 'quality', 'duration', 'speaker', 'condition', 'accented', 'wi', 'wm', 'wf', 'ai', 'am', 'af', 'ii', 'im', 'iif', 'ui', 'um', 'uf', 'bi', 'bi_start', 'prev', 'prev_gem', 'foll', 'foll_gem', 'f1', 'f2', 'f3', 'utt_rate', 'word_rate', 'prevduration', 'follduration', 'istest')
  }
  
  # Create a new column called prevbool/follbool, which is 0 if prev/foll is EOW or a pause, and is 1 otherwise
  data$prevbool <- data$prevduration
  data$follbool <- data$follduration
  data$prevbool[data$prev == 'EOW' | data$prev == '<pz>']  <- 0
  data$prevbool[data$prev != 'EOW' & data$prev != '<pz>']  <- 1
  data$follbool[data$foll == 'EOW' | data$foll == '<pz>'] <- 0
  data$follbool[data$foll != 'EOW' & data$foll != '<pz>']  <- 1
  
  # Set the prevduration/follduration to 0 for cases where you have an EOW or pause as foll/prev
  data <- within(data, prevduration[prev == 'EOW' | prev == '<pz>'] <- 0)
  # data <- within(data, prevratio[prev == 'EOW' | prev == '<pz>'] <- 0)
  data <- within(data, follduration[foll == 'EOW' | foll == '<pz>'] <- 0)
  # data <- within(data, follratio[foll == 'EOW' | foll == '<pz>'] <- 0)
  
  # Get mean of prevduration and follduration (ignoring cases where it's 0)
  prevmean <- mean(data$prevduration[data$prevbool == 1])
  follmean <- mean(data$follduration[data$follbool == 1])
  
  # Set any case where you have an EOW/pause in prev/foll to the mean of the remaining cases
  data$prevduration[data$prevbool == 0] <- prevmean
  data$follduration[data$follbool == 0] <- follmean
  
  # Get rid of prevratio/follratio which I really just should not have been using before
  data$prevratio <- NULL
  data$follratio <- NULL
  
  # Get rid of prevbook and follbool which were just a tool for getting the correct prevduration/folldurations
  data$prevbool <- factor(data$prevbool)
  data$follbool <- factor(data$follbool)
  data$prevbool <- NULL
  data$follbool <- NULL
  
  # This gets us weights
  data$weights[data$length == 0] <- 1
  data$weights[data$length == 1] <- sum(data$length == 0) / sum(data$length == 1)
  return(data)
}

# Shuffle the data (keeping # of short/long vowels consistent)
shuffle_data <- function(data) {
  data <- data[sample(nrow(data)),]
  # Split data into short and long
  short <- subset(data, length == 0)
  long <- subset(data, length == 1)
  
  # Get training and testing
  n_shorttrain <- floor(0.9*nrow(short))
  n_longtrain <- floor(0.9*nrow(long))
  train_short <- short[1:n_shorttrain,]
  test_short <- short[(n_shorttrain+1):nrow(short),]
  train_long <- long[1:n_longtrain,]
  test_long <- long[(n_longtrain+1):nrow(long),]
  
  # Merge into one big train and test file
  train <- rbind(train_short, train_long)
  test <- rbind(test_short, test_long)
  
  # Shuffle the data so that it's not just all the shorts and then all of the longs
  train <- train[sample(nrow(train)),]
  test <- test[sample(nrow(test)),]
  return(list("train"=train, "test"=test))
}

# Linear regression (multi-target = duration & formants)
do_multitarget_linear_regression <- function(targets, inputs, train, test) {
  for(target in targets) {
    form <- paste0(target, ' ~ ', inputs)
    out_model <- lm(form, data=train)
    train[[paste0('residuals_', target)]] <- train[[target]] - my_lm_predict(out_model, train)
    test[[paste0('residuals_', target)]] <- test[[target]] - my_lm_predict(out_model, test)
  }
  return(list("model"=out_model, "train"=train, "test"=test))
}

# Given model, predict results on test set
my_lm_predict <- function(model, test) {
  predictive_vars <- all.vars(formula(model)[[3]])
  coefs <- model$coefficients
  coef_names <- names(coefs)
  scores <- c()
  
  for(i in 1:nrow(test)) {
    score <- coefs[['(Intercept)']]
    for(var in predictive_vars) {
      if(is.factor(test[[var]])) {
        code <- paste0(var, test[[var]][i])
        if(code %in% coef_names) {
          if(!is.na(coefs[[code]])) {
            score <- score + coefs[[code]]
          }
        }
      } else {
        if(!is.na(coefs[[var]])) {
          score <- score + coefs[[var]] * test[[var]][i]
        }
      }
    }
    scores <- c(scores, score)
  }
  return(scores)
}

# Train and run logistic regression model
train_and_predict_glm <- function(form, train, test, use_weights=FALSE) {
  if(use_weights) {
    model <- glm(form, family=binomial(link='logit'), data=train, weights=train$weights)
  } else {
    model <- glm(form, family=binomial(link='logit'), data=train)
  }
  model_bic <- BIC(model)
  
  # Get predictions and accuracy
  fitted.results <- my_glm_predict(model,test)
  fitted.results <- ifelse(fitted.results > 0.5,1,0)
  accuracy <- 1-mean(fitted.results != test$length)
  fitted.results.short <- subset(fitted.results, test$length == 0)
  short.accuracy <- mean(fitted.results.short == 0)
  fitted.results.long <- subset(fitted.results, test$length == 1)
  long.accuracy <- mean(fitted.results.long == 1)
  return(list("model"=model, "BIC"=model_bic, "accuracy"=accuracy, "short.accuracy"=short.accuracy, "long.accuracy"=long.accuracy))
}

# Given GLM, make prediction on test
my_glm_predict <- function(model, test) {
  return(1/(1+exp(-1*my_lm_predict(model, test))))
}

# Calculate BIC
my_bic <- function(model, trainX, trainY, inputs) {
  probs <- predict(model, trainX, s = "lambda.min", type="response")
  probs[trainY == 0] <- 1 - probs[trainY == 0]
  ll <- sum(log(probs + 1e-10))
  bic <- log(nrow(trainX))*length(inputs) - 2*ll
  return(bic)
}
