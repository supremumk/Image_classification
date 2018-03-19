train <- read.csv("internetads_train.csv", as.is=TRUE)
test <- read.csv("internetads_test.csv",as.is=TRUE)
table(sapply(train, class))
table(sapply(test, class))

which(sapply(train, class) == "character")
# We should convert these first four columns into numeric values.
for (i in 1:4) {
  train[,i] <- as.numeric(train[,i])
  test[,i] <- as.numeric(test[,i])
}

# missing
ads <- rbind(train, test) # gluing these together for now
v <- sapply(ads, function(x) sum(is.na(x)))
hist(v)
# It appears that all missing values are found in the first 4 columns

# impute :k-nearest neighbors could be an effective approach if we believed that the missing values appeared at random
#in the dataset

#a bit of digging here. We’ll identify the first row that contains missing values in these entries:
which(is.na(ads[,1]))[1] # row 10

# Which of the other features were identified for this observation (in the url’s , captions, etc.)?
nonzeros <- which(ads[10,-1559] > 0)
ads[10,c(1:3,nonzeros)]

# 5 neighbors
colsFill <- 5:1558
target <- as.vector(ads[10, colsFill])
tmp <- as.matrix(ads[apply(ads[,1:3], 1, function(x) !any(is.na(x))),])
dists <- apply(tmp[,colsFill], 1, function(x) sum(x != target))
closest5 <- head(order(dists),5)
hist(dists)

tmp[closest5,c(1:3, nonzeros)]
# among the five closest images (in Euclidean space), it would appear as though only the 1st 3 images
# are actually related to our target image. The other two have a small Euclidean distance, but the salient
# features in our target image are completely absent in these. This would suggest that despite our having a lot
# of columns at our disposal for imputing via k-nn, the feature space is extremely sparse. We should probably
# use a small number for k in order to make sure we’re not introducing extra bits of noise in this imputation process.

K <- 8
i <- 1 # column to impute
set.seed(665)
tofill <- which(is.na(ads[,i]))
newtrain <- (1:nrow(ads))[-tofill]
s <- sample(newtrain, length(newtrain)*.8, replace=F)
newvalid <- ads[-c(tofill, s),]
newtrain <- ads[s,]
errs <- rep(NA, K)
for (k in 1:K) {
  p_k <- knn.reg(newtrain[,colsFill],
                 newvalid[,colsFill],
                 newtrain[,i],
                 k=k)$pred
  errs[k] <- mean((newvalid[,i] - p_k)^2)
}
plot(errs)
which.min(errs) # 3 seems best

i <- 2
set.seed(665)
tofill <- which(is.na(ads[,i]))
newtrain <- (1:nrow(ads))[-tofill]
s <- sample(newtrain, length(newtrain)*.8, replace=F)
newvalid <- ads[-c(tofill, s),]
newtrain <- ads[s,]
errs <- rep(NA, K)
for (k in 1:K) {
  p_k <- knn.reg(newtrain[,colsFill],
                 newvalid[,colsFill],
                 newtrain[,i],
                 k=k)$pred
  errs[k] <- mean((newvalid[,i] - p_k)^2)
}

plot(errs)
# We don’t have to impute column 3, since that is simply the ratio of the previous 2 columns.
i <- 4
set.seed(665)
tofill <- which(is.na(ads[,i]))
newtrain <- (1:nrow(ads))[-tofill]
s <- sample(newtrain, length(newtrain)*.8, replace=F)
newvalid <- ads[-c(tofill, s),]
newtrain <- ads[s,]
errs <- rep(NA, K)
for (k in 1:K) {
  p_k <- knn.reg(newtrain[,colsFill],
                 newvalid[,colsFill],
                 newtrain[,i],
                 k=k)$pred >= 0.5
  p_k <- p_k * 1
  errs[k] <- mean((newvalid[,i] != p_k))
}
plot(errs)
which.min(errs) # 1 seems best

for (i in c(1:2, 4)) {
  k <- ifelse(i <= 2, 3, 1)
  tofill <- which(is.na(ads[,i]))
  tmpvals <- knn.reg(ads[-tofill,colsFill],
                     ads[tofill,colsFill],
                     ads[-tofill, i],
                     k=k)$pred
  if (i == 4) tmpvals <- (tmpvals >= 0.5)*1 # this is a binary variable
  ads[tofill,i] <- tmpvals
}
# fill in the missing aspect ratios:
tofill <- is.na(ads[,3])
ads[tofill,3] <- ads[tofill,2]/ads[tofill,1]
# separate our ads variable back into the training and the test sets.
train <- ads[1:nrow(train),]
test <- ads[-(1:nrow(train)),]

# Tree with Maximum Depth of 3
r1 <- rpart(class ~ ., data=train, maxdepth=3)
p1 <- predict(r1, test)
p1 <- ifelse(p1[,1] > 0.5, "ad.", "nonad.")
table(predict=p1, actual=test$class)
mean(p1 != test$class)
par(xpd=TRUE)
plot(r1)
text(r1)

# Tree with Maximum Depth of 5
r2 <- rpart(class ~ ., data=train, maxdepth=5)
p2 <- predict(r1, test)
head(p2)
p2 <- ifelse(p2[,1] > 0.5, "ad.", "nonad.")
table(predict=p2, actual=test$class)
mean(p2 != test$class)
par(xpd=TRUE)
plot(r2)
text(r2)

# bagging
set.seed(665)
train$class <- factor(train$class)
r3 <- randomForest(class ~., data=train, mtry = ncol(train)-1, ntree=50,
                   importance = TRUE)
r3

p3 <- predict(r3, test)
table(predict=p3, actual=test$class)

mean(p3 != test$class)

# A variable importance plot
varImpPlot(r3, n.var=10, type=2)
