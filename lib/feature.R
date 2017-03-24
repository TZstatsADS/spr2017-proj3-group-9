
feature <- function(img_dir,img_name,data_name="data", export=T){
  path0 <- paste0(img_dir,"/sift_features.csv",collapse = "")
  features1<-read.csv(path0)
  features2<-t(features1)
  
  train_data<-list()
  for (i in 1:2000){
    n<-nchar(as.character(i))
    path<-paste0(img_dir,"/",img_name,"_",paste(rep(0,4-n),collapse=""),i,".jpg")
    train_data[[i]]<-resize(readImage(path),128,128)
  }
  
  
  
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
  feature0<-unlist(feature0)
  
  
  #non-zero feature
  
  num<-apply(abs(sign(features1)),1,sum)
  
  
  
  # using HOG to get features
  #install.packages("OpenImageR")
  library(OpenImageR)
  
  hog <- vector()
  
  for (i in 1:2000){
    hog <- rbind(hog,HOG(train_data[[i]]))
  }
  
  feature <- cbind(num,feature0,hog)
  
  if(export){
    save(feature,file="../output/featurenew.Rdata")
  }
  
  
  return(feature)
  
}
