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


#Remove outliers based on data based on a control chart method
#Data needs to be scaled
remove_outliers_v2 <- function(x, na.rm = TRUE, ...) {
  H_upper <- 3 * sd(x, na.rm = TRUE)
  H_lower <- -1.5 * sd(x, na.rm = TRUE)
  y <- x
  y[x < (H_lower)- mean(x,na.rm= TRUE)] <- NA
  y[x > (H_upper)+ mean (x,na.rm = TRUE)] <- NA
  y
}


#removes lag assocaited with the prewhiting
lag_function<- function(var,length){
	y=c(tail(var,length(var)-length),rep(NA,length))
	return (y)
}
lead_function<- function(var,length){
	y=c(rep(NA,length),head(var,length(var)-length))
	return (y)
}


am14<-read.csv("C:/Users/Admin/Documents/Wind_Data/AM/AM14/all.txt",header=T)
am14<-am14[,-xtfrm (20:23)]

dat14=as.POSIXct(strptime(paste(am14[,1], sep=" "),
format="%m/%d/%Y %H:%M"),tz="EST")

am14=am14[,-1]
am14<-cbind(dat14,am14)
am14<-am14[order(as.Date(am14[,1], format="%Y-%m-%d")),]

am14<-am14[complete.cases(am14[ , 1:6]),] #remove rows with all NA

y<-am14
y<-y[,-xtfrm (2:6)]


date14<-y[,1] #saving the date vector for visualization
date14<-date14[2:159263] #adjusting for lag
write.table(date14, file = "date14.txt",sep="\t",
 col.names=TRUE,row.names=F)

#Create a gbox reseasonal vector
gbox<-na.interp(y[,7]) 
var<-stl(ts(gbox,frequency=4329), "periodic")
reseason_am14_gbox<-var$time.series[,1]

#Create seasonal values for all features
for (i in 1:14 ){
	if (i == 1) next
	y[,i]<-na.interp(y[,i]) #interpret missing values
	var<-stl(ts(y[,i],frequency=4329), "periodic") # lowess method to detect seasonality
	y[,i]<-as.vector(seasadj(var))
}


scale_y<-scale(y[,2:14])
#attr(scale_y,"scaled:center") 
#attr(scale_y,"scaled:scale") 


scale_y[,3]<-lead_function(scale_y[,3],1) #lag removal based on gbox temp
scale_y[,8]<-lag_function(scale_y[,8],1)
scale_y[,10]<-lag_function(scale_y[,10],1)
scale_y[,13]<-lag_function(scale_y[,13],1)

Gear<-scale_y[,6]
scale_y<-scale_y[,-6]
scale_y<-cbind(scale_y,Gear)

scale_y<-scale_y[complete.cases(scale_y), ] #removing the NA introduced by Lag removal

scale_y_out<-apply(scale_y,2,remove_outliers_v2 )
scale_y_out[is.na(scale_y_out)] <- -4 #masking outliers with a very negative value

write.table(scale_y_out, file = "am14_outliers_seasonal.txt",sep="\t",
col.names=TRUE,row.names=F) #file with outliers masked

write.table(scale_y, file = "am14_seasonal.txt",sep="\t",
col.names=TRUE,row.names=F) #file with outliers

write.table(reseason_am14_gbox, file = "reseason_am14_gbox.txt",sep="\t",
col.names=TRUE,row.names=F) #file reseasonal of gbox temp


