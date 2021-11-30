---
layout: single
title: "BADM Hackathon Technical"
header:
excerpt: "Methodolgy of the 1st place machine learning model"
date: November 29, 2021
---
# BADM Hackathon Technical Components
For the 2021 BADM Hackathon, my team and I placed first for both the technical and storytelling aspects. This post will go through our process on how we came up with the best scoring model. 

# Case Context
Meta-Visit (MV) is a startup company in British Columbia (B.C.) that works in providing virtual care technology to patients that need access to general physicians (GP’s). In B.C., this type of service is covered under the medical services plan (MSP). However, with the exponential growth of digital health solutions around the world, MV wants to expand into the non-MSP environment. Non-MSP are health services that are usually paid out of pocket or by third party insurances. One of the first decisions that needs to be made is how we can target these non-MSP users. This is done by a collection of self-reported data from MV’s current customer base through a questionnaire. With this self-reported data, we are also given geodemographic and user data. The main goal of the technical portion is to combine these datasets and create a statistical/machine learning model that can predict the power user of patients. A power user is defined as a patient that is more likely to use the virtual care technology platform based on a frequency greater than 11. The judging criteria is based on capturing the largest number of power users at the 40% sample proportion level of a lift chart.

# Data Wrangling
{% highlight R %}
#Reading in the data
usage  <- read_csv("USAGE_DATASET.CSV")
survey <- read_csv("SURVEY_DATASET.CSV")
fsa    <- read_csv("FSA_DATASET.CSV")

# clean the first rows as it contained a surrogate key of the row number
survey = survey[,-1]
usage = usage[,-1]
fsa = fsa[,-1]

# pre-checking, find out unique identifier (primary key)
length(unique(survey$pt_id))
length(survey$pt_id)
length(unique(usage$clinic_id))
length(unique(usage$pt_id))
length(usage$pt_id)

# merge the datasets together
full <- survey %>% 
  left_join(usage, "pt_id") %>% 
  left_join(fsa, c("clinic_fsa"="fsa"))
full
full = rename(full, "spend_health"="spending" )
{% endhighlight %}

# Transforming Data Types
{% highlight R %}
# Change the variables into factor/numeric
cols = c("spend_health", "median_age_fsa", "hhold_fsa", "median_income_fsa",
          "hhold_work_health","avg_spend_health", "avg_dcost", "avg_insur_prem", "tot_spend_toba_alco", "freq", "pop_fsa")
full[cols] = lapply(full[cols], as.numeric)
full[,-which(names(full) %in% c(cols, "Sample", "power_us"))] = 
  lapply(full[,-which(names(full) %in% c(cols, "Sample", "power_us"))], as.factor) 
full$power_us.f=as.factor(full$power_us)
{% endhighlight %}

# Creating Variables
These new variables were created based on the current variables given in the three datasets. This section tests our domain knowledge of digital health because these variables may not be intuitive. The statistically significant variables that improved our models were kept and the rest were removed.

{% highlight R %}
# Average household spend on tobacco and alcohol
full$avg_spend_toba_alco = full$tot_spend_toba_alco/full$hhold_fsa

# Household size for each FSA
full$size=full$pop_fsa/full$hhold_fsa 

# How much 'more' money the household of that patient spend compare to avg household
full$spend_difference = (full$spend_health*full$size) - full$avg_spend_health

# Additional created variables that were taken into consideration but ultimately lowered the model scores and were removed
full$spend_difference = full$spend_difference/full$size
full$median_income_fsa = full$median_income_fsa/full$size
full$avg_dcost = full$avg_dcost/full$size
full$avg_spend_toba_alco = full$avg_spend_toba_alco/full$size
{% endhighlight %}

# Dealing with NA’s in the bmi_class and perc_weight variables
{% highlight R %}
full$bmi_class=as.character(full$bmi_class)
full$perc_weight=as.character(full$perc_weight)
full[is.na(full$perc_weight),]$perc_weight = c("Not Reported")
full[is.na(full$bmi_class),]$bmi_class = c("Not Reported")
full$bmi_class=as.factor(full$bmi_class)
full$perc_weight=as.factor(full$perc_weight)
{% endhighlight %}

#Split dataset and check
{% highlight R %}
train = full %>% filter(Sample=="Estimation")
test = full %>% filter(Sample=="Validation")
hold = full %>% filter(Sample=="Holdout")
# check
colnames(hold)[colSums(is.na(hold)) > 0]
colnames(test)[colSums(is.na(test)) > 0]
colnames(train)[colSums(is.na(train)) > 0]
length(train$pt_id)
length(test$pt_id)
{% endhighlight %}

Modelling
All models use the same variables, but with highly correlated and irrelevant variables removed. Matrix scatter plots were used to test correlation. 
# Logistic Regression
model1.logreg <- glm(power_us.f ~ income + age + edu 
                     + perc_health + perc_weight+ bmi_class + arthritis 
                     + highBP + diabetes + stroke + repstrain 
                     + injstatus + physactivityindicator + gave_birth_last5 + perc_mentalHealth 
                     + care_language + othercare 
                     + spend_health 
                     + median_age_fsa 
                     + median_income_fsa+ hhold_work_health  
                     + avg_dcost + avg_insur_prem
                     + avg_spend_toba_alco + spend_difference,
                     data = train, family = binomial(logit))
summary(model1.logreg)
#Estimation - Cumulative

# Stepwise Regression
model2.step <- step(model1.logreg,direction="both")
summary(model2.step)

# Classification and Regression Tree (CART)
model3.rpart <- rpart(formula = power_us.f ~ income + age + edu 
                      + perc_health + perc_weight+ bmi_class + arthritis 
                      + highBP + diabetes + stroke + repstrain 
                      + injstatus + physactivityindicator + gave_birth_last5 + perc_mentalHealth 
                      + care_language + othercare 
                      + spend_health 
                      + median_age_fsa 
                      + median_income_fsa+ hhold_work_health  
                      + avg_dcost + avg_insur_prem
                      + avg_spend_toba_alco + spend_difference,
                      data = train,
                      cp = 0.0001, #set to 0.0001 to check 
                      model = TRUE)
plotcp(model3.rpart)
printcp(model3.rpart)
rpart.plot(model3.rpart,type=1,extra=2,fallen.leaves = FALSE,uniform=TRUE, yes.text="true",no.text="false",cex=0.6,digits=2)

# Running a Random Forest
model4.RF<-randomForest(power_us.f ~ income + age + edu 
                        + perc_health + perc_weight+ bmi_class + arthritis 
                        + highBP + diabetes + stroke + repstrain 
                        + injstatus + physactivityindicator + gave_birth_last5 + perc_mentalHealth 
                        + care_language + othercare 
                        + spend_health 
                        + median_age_fsa 
                        + median_income_fsa+ hhold_work_health  
                        + avg_dcost + avg_insur_prem
                        + avg_spend_toba_alco + spend_difference,
                        data=train,
                        mtry=2, ntree= 200,
                        importance = TRUE)
model4.RF
importance(model4.RF,type = 2)
varImpPlot(model4.RF,type = 1, main = "Importance scale, the higher the better", pch=15)

# Running a Neural Net
model5.Neunet <- Nnet(formula = power_us.f ~ income + age + edu 
                      + perc_health + perc_weight+ bmi_class + arthritis 
                      + highBP + diabetes + stroke + repstrain 
                      + injstatus + physactivityindicator + gave_birth_last5 + perc_mentalHealth 
                      + care_language + othercare 
                      + spend_health 
                      + hhold_work_health  
                      + avg_dcost + avg_insur_prem
                      + avg_spend_toba_alco + spend_difference,
                      data = train,
                      decay = 0.5, # decay parameter
                      size = 4)
model5.Neunet$value
summary(model5.Neunet)
# Lift Charts
lift.chart(modelList = c("model1.logreg"),
           data = train,
           targLevel = "1", 
           trueResp = 0.022,
           type = "cumulative", sub = "Estimation")
#Validation - Cumulative
lift.chart(modelList = c("model1.logreg"),
           data = test,
           targLevel = "1", 
           trueResp = 0.022,
           type = "cumulative", sub = "Validation")
#Estimation - Cumulative
lift.chart(modelList = c("model1.logreg", "model2.step", "model3.rpart", "model4.RF"),
           data = train,
           targLevel = "1", 
           trueResp = 0.022,
           type = "cumulative", sub = "Estimation")
#Validation - Cumulative
lift.chart(modelList = c("model1.logreg", "model2.step", "model3.rpart", "model4.RF"),
           data = test,
           targLevel = "1", 
           trueResp = 0.022,
           type = "cumulative", sub = "Validation")
