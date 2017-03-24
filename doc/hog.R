# using HOG to get features
#install.packages("OpenImageR")
library(OpenImageR)

hog <- vector()

for (i in 1:2000){
  n<-nchar(as.character(i))
  path<-paste0("D:/5243/proj 3/training_data/raw_images/image_",paste(rep(0,4-n),collapse=""),i,".jpg")
  a <- readImage(path)
  hog <- rbind(hog,HOG(a))
}

write.csv(hog,file = "hog.csv")
