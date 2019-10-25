## ----setwd---------------------------------------------------------------
setwd("../R")

## ----packages, results="hide", warning=FALSE, message=FALSE--------------
package.list = c("party", "mlr", "rpart", "rpart.plot")
tmp.install = which(lapply(package.list, require, character.only = TRUE)==FALSE)
if(length(tmp.install)>0) install.packages(package.list[tmp.install], repos = "http://cran.us.r-project.org")
lapply(package.list, require, character.only = TRUE)

## ----read_df-------------------------------------------------------------
df = data.frame(read.csv("data_example.csv"))
head(df, 5)

## ----data_train_test-----------------------------------------------------
train.set = sample(1:nrow(df), size = nrow(df)*0.9, replace = F)
test.set  = setdiff(1:nrow(df),train.set)

## ----task_learner_train--------------------------------------------------
regr.task = makeRegrTask(data = df, target = "y1")
regr.lrn = makeLearner("regr.cforest")
regr.mod = train(regr.lrn, regr.task, subset = train.set)
regr.mod
task.pred = predict(regr.mod, task = regr.task, subset = test.set)
performance(task.pred, measures=list(mse, rsq))
plot(as.data.frame(task.pred)[,c("truth","response")])

## ----get_performancemeasures---------------------------------------------
listMeasures(regr.task)

## ----get_hyperparameters-------------------------------------------------
getParamSet(regr.lrn)

## ----finetuning_hyperparameters1-----------------------------------------
ps = makeParamSet(
  makeIntegerParam("ntree", lower = 1, upper = 100),
  makeIntegerParam("mtry", lower =  1, upper = 10)
)

## ----finetuning_hyperparameters_resol, results="hide", warning=FALSE, message=FALSE----
ctrl = makeTuneControlGrid(resolution = 3)
rdesc = makeResampleDesc("CV", iters = 2)
tune.cforest = tuneParams(regr.lrn, task = regr.task, resampling = rdesc, par.set = ps, control = ctrl)

## ----finetuning_hyperparameters_plot-------------------------------------
plotHyperParsEffect(generateHyperParsEffectData(tune.cforest), x = "ntree", y = "mtry", z = "mse.test.mean",
  plot.type = "heatmap")

tune.cforest$x

## ----finetuning_hyperparameters_bestmodel--------------------------------
regr.lrn.best = setHyperPars(makeLearner("regr.cforest"), ntree = tune.cforest$x$ntree, mtry = tune.cforest$x$mtry)
regr.mod.best = train(regr.lrn.best, regr.task, subset = train.set)

## ----vimp----------------------------------------------------------------
vimp = unlist(getFeatureImportance(regr.mod.best)$res)
barplot(vimp)

## ----pdp-----------------------------------------------------------------
pdp.all = generatePartialDependenceData(regr.mod.best, regr.task, individual = F)
plotPartialDependence(pdp.all)

## ----formula-------------------------------------------------------------
# creating formula
formula.1 = as.formula(paste("y1", paste(colnames(df)[2:length(colnames(df))], collapse=" + "), sep=" ~ ")) 
print(formula.1)

## ----cart----------------------------------------------------------------
# regression
cart.1 = rpart(formula.1, data=df, method="anova",control=rpart.control(minsplit=10, cp=0.001)) 

## ----cart_plot-----------------------------------------------------------
prp(cart.1) # plot with rpart.plot package

## ----cart_cv-------------------------------------------------------------
rsq.rpart(cart.1)	#plot approximate R-squared and relative error for different splits (2 plots). labels are only appropriate for the "anova" method.
cp.best = cart.1$cptable[which(cart.1$cptable[,"xerror"]==min(cart.1$cptable[,"xerror"])),"CP"]

## ----cart_best-----------------------------------------------------------
# regression (y1) and classification (y2)
cart.1.best = rpart(formula.1, data=df, method="anova",control=rpart.control(minsplit=10, cp=cp.best)) 

par(mfrow=c(1,2))
prp(cart.1.best)
prp(cart.1)

## ----ctree---------------------------------------------------------------
ctree.1 = ctree(formula.1, df, control = ctree_control(testtype = "Bonferroni", mincriterion = 0.95, minsplit = 10, minbucket = 7))
plot(ctree.1)

## ----ctree_p-value-------------------------------------------------------
ctree.2 = ctree(formula.1, df, control = ctree_control(testtype = "Univariate", mincriterion = 0.95, minsplit = 10, minbucket = 7))
plot(ctree.2)

