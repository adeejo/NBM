##Author: Adi Joshi
##Date: 3/26/2018
##Purpose: To Preprocess the data for SCADA data,
##		by removing lag,seasonality,scaled, outliers 
##		based on the gearbox temperature variable		

##Input: Raw SCADA data for a turbine

##Output: 4 files
##		1: A seasonalized,scaled and lag adjusted SCADA 
##			with outliers 
##		2: A seasonalized,scaled and lag adjusted SCADA 
##			with masked outliers
##		3: Date vector
##		4: Seasonal Gearbox vector
		 

##User defined function: 
##remove_outliers_v2: removes outliers above 3 SD from mean 
##and below -1.5 SD from mean
##lag_function: make the last values NA and shifts 
##		    data towards the start
##lead_function: make the first values NA and shifts 
##		     data towards the end

library(TSA)
library(fpp)

remove_outliers_v2 <- function(x, na.rm = TRUE, ...) {
  H_upper <- 3 * sd(x, na.rm = TRUE)
  H_lower <- -1.5 * sd(x, na.rm = TRUE)
  y <- x
  y[x < (H_lower)- mean(x,na.rm= TRUE)] <- NA
  y[x > (H_upper)+ mean (x,na.rm = TRUE)] <- NA
  y
}

#removes lag assocaited with the pre-whiting
lag_function<- function(var,length){
	y=c(tail(var,length(var)-length),rep(NA,length))
	return (y)
}
lead_function<- function(var,length){
	y=c(rep(NA,length),head(var,length(var)-length))
	return (y)
}


am69<-read.csv("C:/Users/Admin/Documents/Wind_Data/AM/AM69/all.txt",header=T)
dat69=as.POSIXct(strptime(paste(am69[,1], sep=" "),
format="%m/%d/%Y %H:%M"),tz="EST")

am69=am69[,-1]
am69<-cbind(dat69,am69)

am69<-am69[order(as.Date(am69[,1], format="%Y-%m-%d")),]
am69<-am69[complete.cases(am69[ , 1:6]),]
y<-am69
y<-y[,-xtfrm (2:6)]

date69<-y[,1]
date69<-date69[2:156080]

write.table(date69, file = "date69.txt",sep="\t",
col.names=TRUE,row.names=F)

gbox<-na.interp(y[,7])
var<-stl(ts(gbox,frequency=4329), "periodic")
reseason_am69_gbox<-var$time.series[,1]

for (i in 1:14 ){
	if (i == 1) next
	y[,i]<-na.interp(y[,i])
	var<-stl(ts(y[,i],frequency=4329), "periodic") # lowess method to detect seasonality
	y[,i]<-as.vector(seasadj(var))
	
}

#rev<-as.vector(var$time.series[,1])+y[,2] #reseasonalize

scale_y<-scale(y[,2:14])
#attr(scale_y,"scaled:center") # rescale data
#attr(scale_y,"scaled:scale") # rescale data


scale_y[,3]<-lead_function(scale_y[,3],1)
scale_y[,8]<-lag_function(scale_y[,8],1)
scale_y[,10]<-lag_function(scale_y[,10],1)
scale_y[,13]<-lag_function(scale_y[,13],1)
Gear<-scale_y[,6]
scale_y<-scale_y[,-6]
scale_y<-cbind(scale_y,Gear)



scale_y<-scale_y[complete.cases(scale_y), ]#removing the NA introduced by Lag removal

scale_y_out<-apply(scale_y,2,remove_outliers_v2 )
scale_y_out[is.na(scale_y_out)] <- -4 #maksing NA values

write.table(scale_y_out, file = "am69_outliers_seasonal.txt",sep="\t",
col.names=TRUE,row.names=F)

write.table(scale_y, file = "am69_seasonal.txt",sep="\t",
col.names=TRUE,row.names=F)

write.table(reseason_am69_gbox, file = "reseason_am69_gbox.txt",sep="\t",
col.names=TRUE,row.names=F)





