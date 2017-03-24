library(Rcpp)
library(class)
library(EBImage)
library(e1071)

setwd("~/GitHub/spr2017-proj3-group-9")

img_train_dir <- "./data/training_data/training_data/raw_images/"
label_train <- read.csv("./data/training_data/training_data/labels.csv", header=T)

features1<-read.csv("./data/training_data/training_data/sift_features/sift_features.csv")
features2<-t(features1)

label<-label_train[,1]
result<-knn.cv(features2,label,k=2)
percent<-as.numeric(result==label)
p<-sum(percent)/2000

train_data<-list()
for (i in 1:2000){
  n<-nchar(as.character(i))
  path<-paste0("./data/training_data/training_data/raw_images/image_",paste(rep(0,4-n),collapse=""),i,".jpg")
  train_data[[i]]<-resize(readImage(path),128,128)
}
save(train_data,file="../output/train_2.Rdata")

library(plyr)
train_data1<-llply(train_data,thresh,offset=0.05)
train_data2<-llply(train_data1,opening,kern=makeBrush(1, shape='disc'))

portion_count<-function(data){
  data4<-imageData(data)
  p1<-as.numeric(data4[1:10,]==1)
  p2<-as.numeric(data4[119:128,]==1)
  p3<-as.numeric(data4[,1:10]==1)
  p4<-as.numeric(data4[,119:128]==1)
  por<-100*(sum(p1)+sum(p2)+sum(p3)+sum(p4))/(1280*4)
  return(por)
  
}

fea<-llply(train_data2,portion_count)
feature0<-as.vector(fea)
save(feature0,file="./output/feature0.Rdata")

img_seg1 <- thresh(train_data[[1777]],offset=0.05)
img_seg3 <- opening(img_seg1, kern=makeBrush(1, shape='disc')) #轮廓处理

display(img_seg3)
display(train_data[[44]])
data3<-imageData(img_seg3)
data4<-t(data3)
1是白的 算白点比例
p1<-as.numeric(data4[1:10,]==1)
p2<-as.numeric(data4[119:128,]==1)
p3<-as.numeric(data4[,1:10]==1)
p4<-as.numeric(data4[,119:128]==1)

por<-100*(sum(p1)+sum(p2)+sum(p3)+sum(p4))/(1280*4)