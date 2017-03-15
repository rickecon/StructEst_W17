setwd("C:/Users/Olivia/Desktop/Taxi Project")
set.seed(200)
source('Taxi_Logit_Likelihoods_Utilities.R')
library(stargazer); library(ggplot2)
#### MIDWAY - BEFORE PERIOD
mdw_taxi_before <- read.csv("taxi_MDW_before.csv")
mdw_taxi_before$date<- as.Date(mdw_taxi_before$date)
mdw_taxi_before<- mdw_taxi_before[mdw_taxi_before$date<"2015-11-25",]

mdw_cta_before <- read.csv("cta_mdw_before_for_estimation.csv")
mdw_cta_before$date<- as.Date(mdw_cta_before$date)
mdw_cta_before<- mdw_cta_before[mdw_cta_before$date<"2015-11-25",]
mdw_cta_before$weekend_holiday <- mdw_cta_before$weekend+mdw_cta_before$holiday

taxi_simple_data_mdw_before <- list(mdw_taxi_before$any_rain,mdw_taxi_before$weekend,mdw_taxi_before$avg_temp,
                         as.matrix(subset(mdw_taxi_before,select=c(d1,d2,d3,d4,d5,d7,d8,d9,d10,d11,d12))))
train_simple_data_mdw_before <- list(mdw_cta_before$any_rain,mdw_cta_before$weekend_holiday,mdw_cta_before$avg_temp,mdw_cta_before$rides,
                          as.matrix(subset(mdw_cta_before,select=c(d1,d2,d3,d4,d5,d7,d8,d9,d10,d11,d12))))
init_simple_vals = c(-2.5,0.05,0.5,-0.1,0.0001)
init_simple_vals_months = c(-2.5,0.05,0.5,-0.1,0.0001,runif(11,-0.5,0.5))
likelihood_simple(init_simple_vals,taxi_simple_data_mdw_before,train_simple_data_mdw_before,F)
likelihood_simple(init_simple_vals_months,taxi_simple_data_mdw_before,train_simple_data_mdw_before,T)
simple_mdw_before <- nlm(likelihood_simple,init_simple_vals,data_taxi=taxi_simple_data_mdw_before,data_train=train_simple_data_mdw_before,months=F,hessian=T,gradtol=1e-10)
se_simple_mdw_before <- diag(solve(simple_mdw_before$hessian))^0.5
simple_mdw_before

simple_months_mdw_before<- nlm(likelihood_simple,init_simple_vals_months,data_taxi=taxi_simple_data_mdw_before,data_train=train_simple_data_mdw_before,months=T,hessian=T,gradtol=1e-10)
se_simple_months_mdw_before <- diag(solve(simple_months_mdw_before$hessian))^0.5
simple_months_mdw_before

# implied parameter values for range of temps
test_temps <- c(10,20,30,40,50,60,70,80,90)
for (j in test_temps) { print(j); print(simple_mdw_before$estimate[4]*j + simple_mdw_before$estimate[5]*(j^2)); print(simple_months_mdw_before$estimate[4]*j + simple_months_mdw_before$estimate[5]*(j^2))}

init_vals_b = c(-2.5,0.05,0.5,-0.15,0.002,-0.5)
#init_vals=c(simple$estimate,-0.2)
init_vals_months_b <-c(init_vals,runif(11,-1,1))

taxi_basic_data_mdw_before <- list(mdw_taxi_before$any_rain,mdw_taxi_before$weekend,mdw_taxi_before$avg_temp,mdw_taxi_before$meanprice,
                        as.matrix(subset(mdw_taxi_before,select=c(d1,d2,d3,d4,d5,d7,d8,d9,d10,d11,d12))))
train_basic_data_mdw_before <- list(mdw_cta_before$any_rain,mdw_cta_before$weekend_holiday,mdw_cta_before$avg_temp,mdw_cta_before$mean_price,mdw_cta_before$rides,
                         as.matrix(subset(mdw_cta_before,select=c(d1,d2,d3,d4,d5,d7,d8,d9,d10,d11,d12))))
likelihood(init_vals,taxi_basic_data_mdw_before,train_basic_data_mdw_before,F)
likelihood(init_vals_months,taxi_basic_data_mdw_before,train_basic_data_mdw_before,T)
basic_mdw_before <- nlm(likelihood, init_vals_b,data_taxi = taxi_basic_data_mdw_before,data_train= train_basic_data_mdw_before,months=F,hessian=T,gradtol=1e-12)
basic_mdw_before
se_basic_mdw_before <- diag(solve(basic_mdw_before$hessian))^0.5
basic_months_mdw_before <-nlm(likelihood, init_vals_months_b,data_taxi = taxi_basic_data_mdw_before,data_train= train_basic_data_mdw_before,months=T,hessian=T,gradtol=1e-12)
basic_months_mdw_before
se_basic_months_mdw_before<- diag(solve(basic_months_mdw_before$hessian))^0.5

### MIDWAY AFTER PERIOD
mdw_taxi_after <- read.csv("taxi_MDW_after.csv")
mdw_taxi_after$date<- as.Date(mdw_taxi_after$date)

mdw_cta_after <- read.csv("cta_mdw_after_for_estimation.csv")
mdw_cta_after$date<- as.Date(mdw_cta_after$date)
mdw_cta_after$weekend_holiday <- mdw_cta_after$weekend+mdw_cta_after$holiday

taxi_simple_data_mdw_after <- list(mdw_taxi_after$any_rain,mdw_taxi_after$weekend,mdw_taxi_after$avg_temp,
                                    as.matrix(subset(mdw_taxi_after,select=c(d1,d2,d3,d4,d5,d7,d8,d9,d10,d11,d12))))
train_simple_data_mdw_after <- list(mdw_cta_after$any_rain,mdw_cta_after$weekend_holiday,mdw_cta_after$avg_temp,mdw_cta_after$rides,
                                     as.matrix(subset(mdw_cta_after,select=c(d1,d2,d3,d4,d5,d7,d8,d9,d10,d11,d12))))


likelihood_simple(init_simple_vals,taxi_simple_data_mdw_after,train_simple_data_mdw_after,F)
likelihood_simple(init_simple_vals_months,taxi_simple_data_mdw_after,train_simple_data_mdw_after,T)
simple_mdw_after <- nlm(likelihood_simple,init_simple_vals,data_taxi=taxi_simple_data_mdw_after,data_train=train_simple_data_mdw_after,months=F,hessian=T,gradtol=1e-12,steptol = 1e-3)
se_simple_mdw_after <- diag(solve(simple_mdw_after$hessian))^0.5
simple_mdw_after
simple_mdw_after_alt <- optim(init_simple_vals,likelihood, data_taxi=taxi_simple_data_mdw_after,data_train=train_simple_data_mdw_after,months=F,method='BFGS')

simple_months_mdw_after<- nlm(likelihood_simple,init_simple_vals_months,data_taxi=taxi_simple_data_mdw_after,data_train=train_simple_data_mdw_after,months=T,hessian=T,gradtol=1e-10)
se_simple_months_mdw_after <- diag(solve(simple_months_mdw_after$hessian))^0.5
simple_months_mdw_after
test_temps <- c(20,30,40,50,60,70,80,90) #; print(simple_months_mdw_after$estimate[4]*j + simple_months_mdw_after$estimate[5]*(j^2))
for (j in test_temps) { print(j); print(simple_mdw_after$estimate[4]*j + simple_mdw_after$estimate[5]*(j^2))}


taxi_basic_data_mdw_after <- list(mdw_taxi_after$any_rain,mdw_taxi_after$weekend,mdw_taxi_after$avg_temp,mdw_taxi_after$meanprice,
                                   as.matrix(subset(mdw_taxi_after,select=c(d1,d2,d3,d4,d5,d7,d8,d9,d10,d11,d12))))
train_basic_data_mdw_after <- list(mdw_cta_after$any_rain,mdw_cta_after$weekend_holiday,mdw_cta_after$avg_temp,mdw_cta_after$mean_price,mdw_cta_after$rides,
                                    as.matrix(subset(mdw_cta_after,select=c(d1,d2,d3,d4,d5,d7,d8,d9,d10,d11,d12))))
likelihood(init_vals,taxi_basic_data_mdw_after,train_basic_data_mdw_after,F)
likelihood(init_vals_months,taxi_basic_data_mdw_after,train_basic_data_mdw_after,T)
basic_mdw_after <- nlm(likelihood,init_vals_b,data_taxi = taxi_basic_data_mdw_after,data_train= train_basic_data_mdw_after,months=F,hessian=T,gradtol=1e-12,steptol=1e-5)
basic_mdw_after
se_basic_mdw_after <- diag(solve(basic_mdw_after$hessian))^0.5
basic_months_mdw_after <-nlm(likelihood, init_vals_months_b,data_taxi = taxi_basic_data_mdw_after,data_train= train_basic_data_mdw_after,months=T,hessian=T,gradtol=1e-12)
basic_months_mdw_after
se_basic_months_mdw_after <- diag(solve(basic_months_mdw_after$hessian))^0.5



### all temp effects, plotted:
# implied parameter values for range of temps

test_temps<- seq(from=0,to=100,by=1)
estimators<- list(simple_mdw_before$estimate,simple_months_mdw_before$estimate,basic_mdw_before$estimate,basic_months_mdw_before$estimate,
                  simple_mdw_after$estimate, simple_months_mdw_after$estimate,basic_mdw_after$estimate,basic_months_mdw_after$estimate)
plot(test_temps,estimators[[1]][4]*test_temps + estimators[[1]][5]*(test_temps^2),ylim=c(0,2))
for (j in estimators[2:length(estimators)]) {
  lines(test_temps,j[4]*test_temps + j[5]*(test_temps^2))
}

ggplot()+geom_line(aes(x=test_temps, y=estimators[[3]][4]*test_temps + estimators[[3]][5]*(test_temps^2),color="Before 2016")) +
  geom_line(aes(x=test_temps, y=estimators[[7]][4]*test_temps + estimators[[7]][5]*(test_temps^2),color="After 2016")) + theme_bw()+
  labs(title="Temperature Effects at Midway, without month FEs", x= "Degrees Fahrenheit", y="Implied Utility Shift for Taxis") + 
  labs(colour="") + theme(legend.position="bottom")
ggsave('tempeffects_mdw_nomonth.png', scale=1)
ggplot()+geom_line(aes(x=test_temps, y=estimators[[4]][4]*test_temps + estimators[[4]][5]*(test_temps^2),color="Before 2016")) +
  geom_line(aes(x=test_temps, y=estimators[[8]][4]*test_temps + estimators[[8]][5]*(test_temps^2),color="After 2016")) + theme_bw()+
  labs(title="Temperature Effects at Midway, with month FEs", x= "Degrees Fahrenheit", y="Implied Utility Shift for Taxis") + 
  labs(colour="") + theme(legend.position="bottom")
ggsave('tempeffects_mdw_month.png', scale=1)

### O HARE BEFORE
ord_taxi_before <- read.csv("taxi_ORD_before.csv")
ord_taxi_before$date<- as.Date(ord_taxi_before$date)
ord_taxi_before<- ord_taxi_before[ord_taxi_before$date<"2015-11-25",]

ord_cta_before <- read.csv("cta_ord_before_for_estimation.csv")
ord_cta_before$date<- as.Date(ord_cta_before$date)
ord_cta_before<- ord_cta_before[ord_cta_before$date<"2015-11-25",]
ord_cta_before$weekend_holiday <- ord_cta_before$weekend+ord_cta_before$holiday

taxi_simple_data_ord_before <- list(ord_taxi_before$any_rain,ord_taxi_before$weekend,ord_taxi_before$avg_temp,
                                    as.matrix(subset(ord_taxi_before,select=c(d1,d2,d3,d4,d5,d7,d8,d9,d10,d11,d12))))
train_simple_data_ord_before <- list(ord_cta_before$any_rain,ord_cta_before$weekend_holiday,ord_cta_before$avg_temp,ord_cta_before$rides,
                                     as.matrix(subset(ord_cta_before,select=c(d1,d2,d3,d4,d5,d7,d8,d9,d10,d11,d12))))
init_simple_vals = c(-1.5,0.5,1,-0.005,0.00001)+runif(5,-0.001,0.001)
init_simple_vals_months = c(init_simple_vals,runif(11,-1,1))
likelihood_simple(init_simple_vals,taxi_simple_data_ord_before,train_simple_data_ord_before,F)
likelihood_simple(init_simple_vals_months,taxi_simple_data_ord_before,train_simple_data_ord_before,T)
simple_ord_before <- nlm(likelihood_simple,init_simple_vals,data_taxi=taxi_simple_data_ord_before,data_train=train_simple_data_ord_before,months=F,hessian=T,gradtol=1e-10,steptol=1e-4)
se_simple_ord_before <- diag(solve(simple_ord_before$hessian))^0.5
simple_ord_before

simple_months_ord_before<- nlm(likelihood_simple,init_simple_vals_months,data_taxi=taxi_simple_data_ord_before,data_train=train_simple_data_ord_before,months=T,hessian=T,gradtol=1e-10)
se_simple_months_ord_before <- diag(solve(simple_months_ord_before$hessian))^0.5
simple_months_ord_before

# implied parameter values for range of temps
test_temps <- c(20,30,40,50,60,70,80)
for (j in test_temps) { print(j); print(simple_ord_before$estimate[4]*j + simple_ord_before$estimate[5]*(j^2)); print(simple_months_ord_before$estimate[4]*j + simple_months_ord_before$estimate[5]*(j^2))}

init_vals = c(-5,0.05,0.5,-0.15,0.002,-0.5)
#init_vals=c(simple$estimate,-0.2)
init_vals_months <-c(init_vals,runif(11,-1,1))
taxi_basic_data_ord_before <- list(ord_taxi_before$any_rain,ord_taxi_before$weekend,ord_taxi_before$avg_temp,ord_taxi_before$meanprice,
                                   as.matrix(subset(ord_taxi_before,select=c(d1,d2,d3,d4,d5,d7,d8,d9,d10,d11,d12))))
train_basic_data_ord_before <- list(ord_cta_before$any_rain,ord_cta_before$weekend_holiday,ord_cta_before$avg_temp,ord_cta_before$mean_price,ord_cta_before$rides,
                                    as.matrix(subset(ord_cta_before,select=c(d1,d2,d3,d4,d5,d7,d8,d9,d10,d11,d12))))
likelihood(init_vals,taxi_basic_data_ord_before,train_basic_data_ord_before,F)
likelihood(init_vals_months,taxi_basic_data_ord_before,train_basic_data_ord_before,T)
basic_ord_before <- nlm(likelihood, init_vals,data_taxi = taxi_basic_data_ord_before,data_train= train_basic_data_ord_before,months=F,hessian=T,gradtol=1e-12)
basic_ord_before
basic_months_ord_before <-nlm(likelihood, init_vals_months,data_taxi = taxi_basic_data_ord_before,data_train= train_basic_data_ord_before,months=T,hessian=T,gradtol=1e-12)
basic_months_ord_before


### O HARE AFTER
ord_taxi_after <- read.csv("taxi_ORD_after.csv")
ord_taxi_after$date<- as.Date(ord_taxi_after$date)

ord_cta_after <- read.csv("cta_ord_after_for_estimation.csv")
ord_cta_after$date<- as.Date(ord_cta_after$date)
ord_cta_after$weekend_holiday <- ord_cta_after$weekend+ord_cta_after$holiday

taxi_simple_data_ord_after <- list(ord_taxi_after$any_rain,ord_taxi_after$weekend,ord_taxi_after$avg_temp,
                                    as.matrix(subset(ord_taxi_after,select=c(d1,d2,d3,d4,d5,d7,d8,d9,d10,d11,d12))))
train_simple_data_ord_after <- list(ord_cta_after$any_rain,ord_cta_after$weekend_holiday,ord_cta_after$avg_temp,ord_cta_after$rides,
                                     as.matrix(subset(ord_cta_after,select=c(d1,d2,d3,d4,d5,d7,d8,d9,d10,d11,d12))))
init_simple_vals = c(-1.5,0.5,1,-0.005,0.00001)+runif(5,-0.001,0.001)
init_simple_vals_months = c(init_simple_vals,runif(11,-1,1))
likelihood_simple(init_simple_vals,taxi_simple_data_ord_after,train_simple_data_ord_after,F)
likelihood_simple(init_simple_vals_months,taxi_simple_data_ord_after,train_simple_data_ord_after,T)
simple_ord_after <- nlm(likelihood_simple,init_simple_vals,data_taxi=taxi_simple_data_ord_after,data_train=train_simple_data_ord_after,months=F,hessian=T,gradtol=1e-10,steptol=1e-4)
se_simple_ord_after <- diag(solve(simple_ord_after$hessian))^0.5
simple_ord_after

simple_months_ord_after<- nlm(likelihood_simple,init_simple_vals_months,data_taxi=taxi_simple_data_ord_after,data_train=train_simple_data_ord_after,months=T,hessian=T,gradtol=1e-10)
se_simple_months_ord_after <- diag(solve(simple_months_ord_after$hessian))^0.5
simple_months_ord_after

# implied parameter values for range of temps
test_temps <- c(20,30,40,50,60,70,80)
for (j in test_temps) { print(j); print(simple_ord_after$estimate[4]*j + simple_ord_after$estimate[5]*(j^2)); print(simple_months_ord_after$estimate[4]*j + simple_months_ord_after$estimate[5]*(j^2))}

init_vals = c(-5,0.05,0.5,-0.15,0.002,-0.5)
#init_vals=c(simple$estimate,-0.2)
init_vals_months <-c(init_vals,runif(11,-1,1))
taxi_basic_data_ord_after <- list(ord_taxi_after$any_rain,ord_taxi_after$weekend,ord_taxi_after$avg_temp,ord_taxi_after$meanprice,
                                   as.matrix(subset(ord_taxi_after,select=c(d1,d2,d3,d4,d5,d7,d8,d9,d10,d11,d12))))
train_basic_data_ord_after <- list(ord_cta_after$any_rain,ord_cta_after$weekend_holiday,ord_cta_after$avg_temp,ord_cta_after$mean_price,ord_cta_after$rides,
                                    as.matrix(subset(ord_cta_after,select=c(d1,d2,d3,d4,d5,d7,d8,d9,d10,d11,d12))))

likelihood(init_vals,taxi_basic_data_ord_after,train_basic_data_ord_after,F)
likelihood(init_vals_months,taxi_basic_data_ord_after,train_basic_data_ord_after,T)
basic_ord_after <- nlm(likelihood, init_vals,data_taxi = taxi_basic_data_ord_after,data_train= train_basic_data_ord_after,months=F,hessian=T,gradtol=1e-12)
basic_ord_after
basic_months_ord_after <-nlm(likelihood, init_vals_months,data_taxi = taxi_basic_data_ord_after,data_train= train_basic_data_ord_after,months=T,hessian=T,gradtol=1e-12)
basic_months_ord_after



