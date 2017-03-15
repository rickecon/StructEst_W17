e_utility_taxi_simple<- function(params,data,months) {
  alpha <- params[1]; a_rain<-params[2]; a_weekend<-params[3]
  b_temp<-params[4]; b_temp2<-params[5]
  # just intercepts,rain, weekend OR holiday,temperature,temperature squared
  # data: a list of rain dummy, weekend or holiday dummy, temperature dummy
  rain<- data[[1]]; weekend_hol <- data[[2]]; temperature<-data[[3]]; temp2 <- data[[3]]^2
  if (months==F){ utility = alpha + a_rain*rain + a_weekend*weekend_hol + b_temp*temperature + b_temp2*temp2 }
  if (months==T){
    a_months<-params[6:16];  month_data <-  data[[4]] # should be NX11
    utility = alpha + a_rain*rain + a_weekend*weekend_hol + b_temp*temperature + b_temp2*temp2 + month_data%*%a_months
  }
  return(utility)
}

likelihood_simple <- function(params,data_taxi,data_train,months) {
  if (months==F){
    u_taxi_taxi<- e_utility_taxi_simple(params,data_taxi,months)
    u_taxi_train <- e_utility_taxi_simple(params,data_train[1:3],months)
    rides<- data_train[[4]]
  }
  if (months==T){
    u_taxi_taxi <-e_utility_taxi_simple(params,data_taxi,months)
    u_taxi_train <- e_utility_taxi_simple(params,data_train[-4],months)
    rides <- data_train[[4]]
  }
  p_taxis <- exp(u_taxi_taxi)/(1+exp(u_taxi_taxi))
  p_train <- 1/(1+ exp(u_taxi_train))
  l_taxis <- sum(log(p_taxis))
  l_train <- sum(rides*log(p_train))
  ll <- l_taxis+l_train
  return(-ll)
}


e_utility_taxi <- function(params,data,months){
  alpha <- params[1]; a_rain<-params[2]; a_weekend<-params[3]
  b_temp<-params[4]; b_temp2<-params[5]; b_price <- params[6]
  rain<- data[[1]]; weekend_hol <- data[[2]]; temperature<-data[[3]]; temp2 <- data[[3]]^2; price<- data[[4]]
  if (months==F){
    utility = alpha + a_rain*rain + a_weekend*weekend_hol + b_temp*temperature + b_temp2*temp2+ b_price*price
  }
  if (months==T) {
    a_months<- params[7:17]; month_data<- data[[5]]
    utility = alpha + a_rain*rain + a_weekend*weekend_hol + b_temp*temperature + b_temp2*temp2+ b_price*price + month_data%*%a_months
  }
  return(utility)
}

likelihood <- function(params,data_taxi,data_train,months) {
  if (months==F) {
    u_taxi_taxi<- e_utility_taxi(params,data_taxi,months)
    u_taxi_train <- e_utility_taxi(params,data_train[1:4],months)
  }
  if (months==T) {
    u_taxi_taxi<- e_utility_taxi(params,data_taxi,months)
    u_taxi_train <- e_utility_taxi(params,data_train[-5],months)
  }
  rides <- data_train[[5]]
  p_taxis <- exp(u_taxi_taxi)/(1+exp(u_taxi_taxi))
  p_train <- 1/(1+ exp(u_taxi_train))
  l_taxis <- sum(log(p_taxis))
  l_train <- sum(rides*log(p_train))
  ll <- l_taxis+l_train
  return(-ll)
}


e_utility_taxi_quality <- function(params, data,months){
  alpha <- params[1]; a_rain<-params[2]; a_weekend<-params[3]
  b_temp<-params[4]; b_temp2<-params[5]; b_price <- params[6]; b_time <- params[7]
  rain<- data[[1]]; weekend_hol <- data[[2]]; temperature<-data[[3]]; temp2 <- data[[3]]^2; price<- data[[4]]; time_savings <- data[[5]]
  if (months==F){
    utility = alpha + a_rain*rain + a_weekend*weekend_hol + b_temp*temperature + b_temp2*temp2+ b_price*price + b_time*time_savings
  }
  if (months==T) {
    a_months<- params[8:18]; month_data<- data[[6]]
    utility = alpha + a_rain*rain + a_weekend*weekend_hol + b_temp*temperature + b_temp2*temp2+ b_price*price + b_time*time_savings + month_data%*%a_months
  }
  return(utility)
}
  
likelihood_quality <- function(params, data_taxi, data_train,months) {
  if (months==F) {
    u_taxi_taxi<- e_utility_taxi_quality(params,data_taxi,months)
    u_taxi_train <- e_utility_taxi_quality(params,data_train[1:5],months)
  }
  if (months==T) {
    u_taxi_taxi<- e_utility_taxi_quality(params,data_taxi,months)
    u_taxi_train <- e_utility_taxi_quality(params,data_train[-6],months)
  }
  rides <- data_train[[6]]
  p_taxis <- exp(u_taxi_taxi)/(1+exp(u_taxi_taxi))
  p_train <- 1/(1+ exp(u_taxi_train))
  l_taxis <- sum(log(p_taxis))
  l_train <- sum(rides*log(p_train))
  ll <- l_taxis+l_train
  return(-ll)
}
  