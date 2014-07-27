
library(caret)
library(ggplot2)

#read the (training) data 
dt<-read.csv("pml-training.csv", na.strings = c("NA",""))
dim(dt)
names(dt)
levels(dt$user_name)
str(dt)

##there are lots of null values and non-numeric columns
## we shall remove them so we can find correlations and 
## later do PCA on them

#A function to remove columsn with null values, 
#and retain only numeric columns
# we shall use it on our training and  testing data provided 
prepareData<-function(dat)
{ 
  colsWithNulls<-unique(which(is.na(dat),arr.ind = T)[,2])
  names(dat[colsWithNulls])
  
  #  let's drop them and see 
  dat2<-dat[,-colsWithNulls]
  dim(dat)
  dim(dat2)
  #let's retain  numeric only columns
  nums <- sapply(dat2, is.numeric)
  nums  
  length(nums==T)
  dat2_n<-dat2[nums]
  
  return(dat2_n)
}


#USe the above to clean up the data
dt2_n<-prepareData(dt)
#class column is stripped.add it back
dt2_n$classe<-dt$classe
dim(dt2_n) # we now have only 57 columsn left



# A quick check on corrlation between columns
M<-abs(cor(dt2_n[,-ncol(dt2_n)]))
diag(M)<-0
which(M>0.8,arr.ind = T)
plot(dt2_n[,22],dt2_n[,23])
plot(dt2_n[,25],dt2_n[,28])

#therefore some features are correlated 
# we will use PCA in our train function to deal with this


# initial runs of random forest were too slow
# therefore trying some things to speed things up
#Use than one processor core to  
library(doMC)
registerDoMC(cores = 2)


# Split the proided training data. 
# keep a chink away for testing

#let's take some 60% of rows instead of 19,622
# leave rest for pure testing / validation 
# maybe we should make a more judicious selection (like splitting by user)
# but I am just using random sampling 

rows<-nrow(dt2_n)
dt2_n_insample<-sample(1:rows,size = round(0.6*rows),replace = F)
str(dt2_n_insample)

dt2_n_Train<-dt2_n[dt2_n_insample,] #this we train on (by further splitting)
dt2_n_Validn<-dt2_n[-dt2_n_insample,] # this we only use once, for testing 

rm(dt2_n_insample) # remove from autocomplete
dim(dt2_n_Train)
dim(dt2_n_Validn)

#further split the training data to training and testing set
inTrain<- createDataPartition(y = dt2_n_Train$classe,
                              p=0.75,list=F)
training<-dt2_n_Train[inTrain,]
testing<-dt2_n_Train[-inTrain,]
dim(testing)
dim(training) #thus we have ~8000 rows to train on.
# This is much faster than using 19,000 rows



# A function to create a model for testing
# we provide it the testing method to use
# and it trains the model, tests it on testing data,
# prints some data 
# regarding its accuracy 
# and retunrs the model
applyMethod<-function(meth)
{
  print("======training, method is====")
  print(meth)
  #we shall restrict it to 15 folds and 5 repeats
  tc<-trainControl(method ="cv",number = 10,repeats = 5 )
  modFit<-train(classe ~ ., trControl = tc,
                method=meth, preProcess="pca", data = training   )
  #summary(modFit$finalModel)
  pred<-predict(modFit, testing)
  conf<-confusionMatrix(pred,testing$classe )
  print( conf$overall )
  print("xxxxxxxxxxend train xxxxxxxxxxx")
  return(modFit)
}

#a function to test the model on validation/testing data that we 
# had set aside . This will be tested only once at the end
testPred<-function(mod)
{
  pred<-predict(mod, dt2_n_Validn)
  conf<-confusionMatrix(pred,dt2_n_Validn$classe )
  print("*********test results *******")
  print( conf$overall )
  print("********* *********** *******")
  return(pred)
}

# A function re predict values based 
# on our model and some given unknown data
# used for final prediction of answers for the assignment
predictValues<-function(mod,testdata)
{
  prd<-predict(mod, testdata)
  print(prd)
  return (prd)
}

# now for the main thing. Fit a model  
mod<-applyMethod("rf")   #  rf, nb, lda
print(mod$method)
#find names of pca's used
print(mod$finalModel$xNames) # therefore, 27 pca's were used to predict
# test on the data we had set aside for validation  
testPred(mod)


#expand on the above
confusionMatrix(predictValues(mod,dt2_n_Validn), dt2_n_Validn$classe)

# we're done with the test data provided 



#now predict for given csv file for testing 
dt_to_predict<-read.csv("pml-testing.csv")
dim(dt_to_predict)
setdiff(names(dt),names(dt_to_predict))
setdiff(names(dt_to_predict), names(dt) )

names(dt_to_predict)
length(complete.cases(dt_to_predict)) # no nulls!
# prepare this smilar to test data
dt3<-prepareData(dt_to_predict)
dim(dt3) #we're down to 57.

#now predict the class of this data
# this is the answer to be submitted
pred<-predictValues(mod,dt3[,-ncol(dt3)]) #remoe the problem_id column


################# END ###################
