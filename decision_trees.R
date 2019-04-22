############################### Decision Trees ###############################

### Training and testing for decision tree models using only PC-loadings extracted 
## from just the pixels as features ####


### Data Setup

boats = read_csv('boats_pixels_blue.csv')

#Remove class labels prior to PCA
boats_labels = boats$type
boats_pixels = boats %>% select(-type)
pca.out = prcomp(boats_pixels, scale = T)

#Add boat type class labels back as "class"
pca.result = data.frame(class = boats_labels, pca.out$x[,1:40]) # based on PVE from before.

#Separate into 80% train, 20% validate
set.seed(2)
train_ids <- sample((1:nrow(boats)), size = floor(.80*nrow(boats)))
train = pca.result[train_ids,]
test = pca.result[-train_ids,]
table(train$class)

set.seed(2)
sub_train_ids = sample(1:nrow(train), size = floor(0.8*nrow(train)))
sub_train = train[sub_train_ids, ]
validate = train[-sub_train_ids, ]

#### DECISION TREE 

### Model building 

## Using tree package
library(tree)

# basic decision tree 
names(train)
tree1 = tree(class~., data=sub_train)
summary(tree1)
plot(tree1)
text(tree1, pretty = 0)

tree.pred = predict(tree1, validate, type="class")

with(validate, table('predictions' = tree.pred, 'actual' = class))
(12+22+10)/nrow(validate)

# pruning tree using cross validation 
set.seed(2)
prune_tree1 = cv.tree(tree1, FUN = prune.misclass, K = 5) # default is 5-fold cross validation 
plot(prune_tree1) # choose 9
pruned = prune.misclass(tree1, best = 9)
plot(pruned)
text(pruned, pretty = 0)
tree.pred = predict(pruned, validate, type="class")

with(validate, table('predictions' = tree.pred, 'actual' = class))
(10+16+17)/nrow(validate) # slightly worsened classification rate 

## Using RPart package 
library(rpart)
tree2 = rpart(class~., data=train)

library(rpart.plot)
rpart.plot(tree2)
output_table = tree2$cptable %>% as.data.frame()
ggplot(output_table, aes(CP, xerror))+geom_point()+geom_line()+
  geom_vline(aes(xintercept = output_table[which.min(output_table$xerror), 'CP']), col = 'red')

thresh = output_table[which.min(output_table$xerror), ]$xerror+output_table[which.min(output_table$xerror), ]$xstd

output_table %>% filter(xerror <= thresh) # choose CP of 0.0625

pruned_tree = prune(tree2, cp = 0.06)
rpart.plot(pruned_tree)

#### RANDOM FOREST 

library(randomForest)
library(caret)
library(e1071)
rf = randomForest(class~., data = sub_train, ntree = 500)
rf

# picking num variables to choose at each split (mtry) using cross validation (4-fold)

oob.err = double(ncol(train)-1)
test.err = double(ncol(train)-1)
for(mtry in 1:(ncol(train)-1)){
  split_df = split(sub_train, c(1:4))
  oob = NA
  test.er = NA
  for (i in 1:length(split_df)) {
    TEST = split_df[[i]]
    train_id = which(i != 1:length(split_df))
    TRAIN = data.frame()
    for (j in 1:length(train_id)) {
      TRAIN = rbind(TRAIN, split_df[[train_id[j]]])
    }
    
    fit = randomForest(class~., data = TRAIN, mtry=mtry)
    oob[i] = (mean(fit$err.rate[, 1]))^2 # mean OOB classication error 
    pred = predict(fit, TEST)
    test.er[i] = sum(pred != validate$class) / nrow(validate) # validation set classification error 
  }
  oob.err[mtry] = (mean(oob))^2
  test.err[mtry] = mean(test.er)
  
}

results = data.frame('mtry' = 1:20, 'oob.error' = oob.err, 'test.err' = test.err)
ggplot(results, aes(mtry))+geom_line(aes(y = oob.error), col = 'red')+
  geom_line(aes(y = test.err))+ylab('classification error')+
  geom_vline(aes(xintercept = results[which.min(results$oob.error), 'mtry']), linetype = 'dashed', col = 'red')+
  geom_vline(aes(xintercept = results[which.min(results$test.err), 'mtry']), linetype = 'dashed')

# choose based off minimum of Out of Bag Error and Test Error
# choose 16 variables at every split 


rf2 = randomForest(class~., data = sub_train, mtry = 16, ntree = 500)
rf2

preds = predict(rf2, validate)
sum(preds == validate$class) / nrow(validate)

## Another shot at tuning


# for MTry and Num Trees 
control <- trainControl(method="oob", number=5,search="grid")
tunegrid <- expand.grid(.mtry=c(1:(ncol(sub_train)-1)))
modellist <- list()
for (ntree in c(100, 250, 500, 750, 1000)) {
  set.seed(2)
  fit <- caret::train(class~., data=sub_train, method="rf", 
                      metric='Accuracy', tuneGrid=tunegrid, trControl=control, ntree=ntree)
  key <- toString(ntree)
  modellist[[key]] <- fit
}

# compare results
avg = c()
std = c()
med = c()
max_acc = c()
best_mtry = c()
for (i in 1:length(modellist)) {
  avg[i] = mean(modellist[[i]]$results$Accuracy)
  std[i] = sd(modellist[[i]]$results$Accuracy)
  med[i] = median(modellist[[i]]$results$Accuracy)
  max_acc[i] = max(modellist[[i]]$results$Accuracy)
  best_mtry[i] = modellist[[i]]$results[which.max(modellist[[i]]$results$Accuracy),'mtry']
}

result = cbind(avg, std, med, max_acc, best_mtry, 
               'ntree' = c(100, 250, 500, 750, 1000))


grid.arrange(ggplot(result %>% as.data.frame(), aes(ntree, max_acc))+geom_line(), 
             ggplot(result %>% as.data.frame(), aes(best_mtry, max_acc))+geom_line())


# choose mtry = 12, ntree = 750
rf3 = randomForest(class~., data = sub_train, mtry = 12, ntree = 750)
rf3

preds = predict(rf3, validate)
sum(preds == validate$class) / nrow(validate)


#### BOOSTING (Using XGBoost)
library(mlr)
library(xgboost)

# converting class labels to numeric as per tutorial
sub_train_boost = sub_train
sub_train_boost$class = as.numeric(sub_train_boost$class)
validate_boost = validate
validate_boost$class = as.numeric(validate$class)

# converting train and validate sets to matrix as per tutorial 
sub_train_boost = as.matrix(sub_train_boost)
validate_boost = as.matrix(sub_train_boost)

train_class = sub_train_boost[, 1]
test_class = validate_boost[, 1]
train_class <- as.numeric(train_class)-1
test_class <- as.numeric(test_class)-1
sub_train_boost = sub_train_boost[, -1]
validate_boost = validate_boost[, -1]


dtrain <- xgb.DMatrix(data = sub_train_boost,label = train_class) 
dtest <- xgb.DMatrix(data = validate_boost,label=test_class)

# using default parameters first as a baseline 
params <- list(booster = "gbtree", objective = "multi:softmax", num_class = 3, 
               eta=0.3, gamma=0, max_depth=6, min_child_weight=1, 
               subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, 
                 nfold = 5, showsd = T, stratified = T, print_every_n = 10, 
                 early_stop_round = 20, maximize = F)

1-min(xgbcv$evaluation_log$test_merror_mean) # baseline classification rate 

ggplot(xgbcv$evaluation_log, aes(iter, test_merror_mean))+geom_line()+xlab('NumTrees')+
  geom_vline(aes(xintercept = which.min(xgbcv$evaluation_log$test_merror_mean)), col = 'red', linetype = 'dashed')

# choose nrounds = 39
opt_nrounds = which.min(xgbcv$evaluation_log$test_merror_mean)

# tuning parameters 

#create tasks
traintask <- makeClassifTask (data = sub_train,target = "class")
testtask <- makeClassifTask (data = validate,target = "class")

#create learner
lrn <- makeLearner("classif.xgboost", predict.type = "response")
lrn$par.vals <- list(objective = "multi:softmax", num_class = 3, eval_metric="mlogloss", 
                     nrounds = opt_nrounds)

#set parameter space
params <- makeParamSet(makeDiscreteParam("booster",values = "gbtree"), 
                       makeIntegerParam("max_depth",lower = 3L,upper = 10L), 
                       makeNumericParam("min_child_weight",lower = 1L,upper = 10L), 
                       makeNumericParam("eta",lower = 0.001,upper = .1), 
                       makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))

#set resampling strategy
rdesc <- makeResampleDesc("CV",stratify = T,iters=5L)

#search strategy
ctrl <- makeTuneControlRandom(maxit = 10L)

#set parallel backend
library(parallel)
library(parallelMap) 
parallelStartSocket(cpus = detectCores())

#parameter tuning
mytune <- tuneParams(learner = lrn, task = traintask, 
                     resampling = rdesc, measures = acc, par.set = params, 
                     control = ctrl, show.info = T)

#set hyperparameters
lrn_tune <- setHyperPars(lrn,par.vals = mytune$x)

#train model
xgmodel <- train(learner = lrn_tune,task = traintask)


#predict model
xgpred <- predict(xgmodel,testtask, type = 'response')
confusionMatrix(xgpred$data$response,xgpred$data$truth)


######################## TESTING MODELS ON HELD OUT TEST SET 

### Basic Decision Tree

set.seed(2)
tree_final = tree(class~., data=train)
summary(tree_final)
plot(tree_final)
text(tree_final, pretty = 0)

tree.pred = predict(tree_final, test, type="class")

with(test, table('predictions' = tree.pred, 'actual' = class))
(11+12+26)/nrow(test)

### Decision Tree with pruning, nodes = 9
pruned = prune.misclass(tree_final, best = 9)
plot(pruned)
text(pruned, pretty = 0)
tree.pred = predict(pruned, test, type="class")

with(test, table('predictions' = tree.pred, 'actual' = class))
(12+15+27)/nrow(test) # slightly worsened classification rate 

### Random Forest: mtry = 12, ntree = 750

rf_final = randomForest(class~., data = train, mtry = 12, ntree = 750)
rf_final

preds = predict(rf_final, test)
sum(preds == test$class) / nrow(test)

### XGBoost: ntrees = 39, learning rate = 0.0442338, num split = 6

# converting train and validate sets to matrix as per tutorial 
train_boost = train
train_boost$class = as.numeric(train_boost$class)

test_boost = test
test_boost$class = as.numeric(test_boost$class)

train_boost = as.matrix(train_boost)
test_boost = as.matrix(test_boost)



train_class = train_boost[, 1]
test_class = test_boost[, 1]
train_class <- as.numeric(train_class)-1
test_class <- as.numeric(test_class)-1
train_boost = train_boost[, -1]
test_boost = test_boost[, -1]


dtrain <- xgb.DMatrix(data = train_boost,label = train_class) 
dtest <- xgb.DMatrix(data = test_boost,label=test_class)

params <- list(booster = "gbtree", objective = "multi:softmax", num_class = 3, eta=0.0442338, 
               gamma=0, max_depth=8, min_child_weight=7.935401,
               subsample=1, colsample_bytree=0.9909192)

xgb1 <- xgb.train (params = params, data = dtrain, nrounds = opt_nrounds, 
                   watchlist = list(val=dtest,train=dtrain), print_every_n = 10, 
                   early_stop_round = 10, maximize = F , eval_metric = "mlogloss")
#model prediction
xgbpred <- predict (xgb1,dtest)
table(xgbpred, test_class)
(25+19+22)/nrow(test)

