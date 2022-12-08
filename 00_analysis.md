Cancer prediction
================
*Nicoló Foppa Pedretti*

### Load libraries

Here I load the libraries used for the analysis

``` r
library(data.table)
library(caret)
library(randomForest)
library(parallel)
library(stringr)
```

### Load and clean data

Here I load the `clinical.csv` and `genomics.csv` datasets:

``` r
DT = fread("clinical.csv")
genomics = fread("genomics.csv")
```

Let’s start with the main task: build and evaluate a predictive model of
one-year survival after diagnosis with NSCLC. In order to answer this
question I create a new variable. The main idea is create a predictive
model after 12 months for dead/alive classification. My assumption is to
create a new outcome value where I set to `Alive` the people that have a
follow-up months longer than 12 months and the Outcome is `Dead`. That
is a strong assumption because technically we don’t have any date of
death (which it would allow us to create this model). I map the binary
like this:

- `Alive` ⟶ 1
- `Dead` ⟶ 0

``` r
DT[,Outcome_num := factor(ifelse(Survival.Months > 12 & Outcome == "Dead",1,0))]
```

I used `Outcome` and `Survival.Months` to create my new outcome that it
will be used for the prediction. We are not using those variables
anymore. Let’s check the other variables

- `Age`: *The patient’s age (in years) at diagnosis*. Numerical
  variable, we don’t need any transformation or cleaning. No missing
  data

- `Grade`: *Tumor grade (1-4 or missing)*.

``` r
table(DT$Grade)
```

    ## 
    ##  2  3  4  9 
    ## 29 22 43 96

Using the description I can assume that the value *9* is referred to
missing data. In this specific case I don’t have any value `1`, but it
could mean that those value are mixed in the code `9` and I cannot use
any imputation on this variable because I don’t have *missing at
random*. Let’s transform the variable in `factor`

``` r
DT[, Grade := ifelse(Grade == 9, NA, Grade)]
DT[, Grade := factor(Grade, exclude = NULL)]
```

- `Num.Primaries`: *Number of primary tumors*. Let’s convert to factor.
  No missing data.

``` r
DT[, Num.Primaries := factor(Num.Primaries, exclude = NULL)]
```

- `T`: *Tumor Stage*. In this variable we have 62 `"UNK"` (Unknown) data
  and a few class with a low sample size

``` r
table(DT$`T`)
```

    ## 
    ##   1  1a  1b   2  2a  2b   3   4 UNK 
    ##   1  26   2  12  16  10  38  23  62

I decide to re-map the (1,1a,1b) ⟶ 1, (2,2a,2b) ⟶ 2 and UKN ⟶ `NA`

``` r
DT[,`T` := factor(ifelse(`T` == "UNK",NA,
                  ifelse(`T` %like% "1",1,
                  ifelse(`T` %like% "2",2,
                  ifelse(`T` %like% "3",3,4)))),exclude = NULL)]
```

- `N`: *Number of metastasis to lymph nodes*. In this variable we have
  65 `"NULL"` (NULL) data and a few class with a low sample size

``` r
table(DT$N)
```

    ## 
    ##    0    1    2    3 NULL 
    ##   52    9   58    6   65

I collapse together classes `0`+`1` and `2`+`3` and I map `"NULL"` in
`NA`

``` r
DT[, N := factor(ifelse(N %in% c("0","1"),"01",
                 ifelse(N %in% c("2","3"),"2+",NA)),exclude = NULL)]
```

- `M`: *Number of distant metastases*. In this variable we have 96
  `"NULL"` variable and only two left unbalanced categories. I convert
  to factor:

``` r
DT[, M := factor(M,exclude = NULL)]
```

- `Radiation`: *Whether radiation took place (5) or not (0)*. No missing
  value and the two class are unbalanced. I map 0/5 in No/Yes and
  convert to a factor:

``` r
DT[, Radiation := factor(ifelse(Radiation == 0,"No","Yes"),exclude = NULL)]
```

- `Stage`: *Stage at diagnosis*. This variable doesn’t have missing data
  but have problems on sample size per category and (possibly) errors on
  the labels (`1B` vs. `IB`). I drop the letters (`A`,`B`) and collapse
  the category based on the numeric stage only:

``` r
table(DT$Stage, useNA = "ifany")
```

    ## 
    ##   1B   IA   IB  IIA  IIB IIIA IIIB   IV  IVB 
    ##    1   32    1    8   11   43   24   45   25

``` r
DT[, Stage := factor(ifelse(Stage %like% "IV","IV",
                     ifelse(Stage %like% "III","III",
                     ifelse(Stage %like% "II","II","I"))))]
```

- `Primary.Site`: *Location of primary tumor*. In this variable I don’t
  have any missing data but I have some subclasses with few
  observations. I simply transform to factor the classes. Having more
  knowledges on lungs functionality I could collapse similar regions.

``` r
DT[, Primary.Site := factor(Primary.Site)]
```

- `Histology`: *The tumor histology*. In this variable I don’t have any
  missing data, I simply transform to factor the classes.

``` r
DT[, Histology := factor(Histology)]
```

- `Tumor.Size`: *Size of the tumor at diagnosis*. Continuous variable
  with a lot of missing data called `"NULL"`. I think that this variable
  could be really important. Let’s check the effect with a single logit
  regression:

``` r
DT$Tumor.Size = as.numeric(DT$Tumor.Size)
```

    ## Warning: NAs introduced by coercion

``` r
summary(glm(DT$Outcome_num~DT$Tumor.Size, 
            family = binomial(link = "logit")))$coeff
```

    ##                 Estimate Std. Error   z value   Pr(>|z|)
    ## (Intercept)   0.05025831  0.4186612 0.1200453 0.90444727
    ## DT$Tumor.Size 0.26138621  0.1019440 2.5640169 0.01034685

I have a lot of missing data but the coefficient is significant. I
wouldn’t impute this value (too many data to impute I can introduce some
biases in the dataset) but I keep it

- `Num.Mutations` and `Num.Mutated.Genes`: *The total number mutations
  found in the tumor and the total number of genes with mutation*. Those
  two variables are highly correlated because the number of mutations of
  each mutated gene is going from 1 to 3. Individually they don’t have
  any significance:

``` r
summary(glm(DT$Outcome_num~DT$Num.Mutated.Genes, 
            family = binomial(link = "logit")))$coeff
```

    ##                         Estimate Std. Error    z value  Pr(>|z|)
    ## (Intercept)           0.22355531 0.30519058  0.7325105 0.4638570
    ## DT$Num.Mutated.Genes -0.04399965 0.09983345 -0.4407306 0.6594081

``` r
summary(glm(DT$Outcome_num~DT$Num.Mutations, 
            family = binomial(link = "logit")))$coeff
```

    ##                    Estimate Std. Error   z value  Pr(>|z|)
    ## (Intercept)       0.4352638 0.30503232  1.426943 0.1535963
    ## DT$Num.Mutations -0.1067785 0.08662863 -1.232601 0.2177248

Let’s create a ratio for the mutation. The coefficient of the ratio in
logit regression is significant

``` r
DT[, ratio_mut := ifelse(Num.Mutated.Genes > 0,
                         Num.Mutations/Num.Mutated.Genes,
                         1)]

summary(glm(DT$Outcome_num~DT$ratio_mut, 
            family = binomial(link = "logit")))$coeff
```

    ##               Estimate Std. Error   z value   Pr(>|z|)
    ## (Intercept)   1.287251  0.5631686  2.285730 0.02227006
    ## DT$ratio_mut -1.016606  0.4735726 -2.146674 0.03181924

I keep only the `ratio_mut` variable for the analysis.

### Chi-square test

We want to test the correlation between the categorical variables and
the outcome. Let’s use a $\chi^2$ test to check the independency of the
two variables:

``` r
cat_variable = names(DT)[c(5,6,7,8,9,10,11,12,13)]
chisq_table = lapply(cat_variable,function(i){
  data.table(Variable = i,
             p_value = round(chisq.test(DT$Outcome_num,DT[[paste0(i)]])$p.value,4))
})
chisq_table = rbindlist(chisq_table)
chisq_table
```

    ##         Variable p_value
    ## 1:         Grade  0.0001
    ## 2: Num.Primaries  0.5165
    ## 3:             T  0.0000
    ## 4:             N  0.0081
    ## 5:             M  0.0000
    ## 6:     Radiation  0.0001
    ## 7:         Stage  0.0001
    ## 8:  Primary.Site  0.0000
    ## 9:     Histology  0.0000

We reject the hypothesis of independency for all variables except for
the `Num.Primaries`. We exclude the variables from the analysis.

### Genomics dataset

In order to include the genomics information into the analysis I pivot
the long table of genomics to wide format to merge clinical dataset and
genomic one. I also transform the variables into factors:

``` r
genomics = dcast(genomics, ID ~ Gene, 
                 value.var = "Gene", 
                 fun.aggregate = length)
genomics[,c(names(genomics)[-1]) := lapply(.SD,function(i) factor(i,exclude = NULL)),
         .SDcols = names(genomics)[-1]]
```

### Merge the datasets

``` r
DT = merge(DT[,.(ID,Age,Grade,`T`,N,M,Radiation,Stage,Primary.Site,
                 Histology,Tumor.Size,Outcome_num,ratio_mut)],
           genomics, by = "ID", all.x = T)
```

### Model

I use a **Random Forest** algorithm to predict the outcome. I pick this
algorithm because is powerful (mean estimation across multiple
classification trees), fast and it can also consider *non-linearity*. I
will include all the dummy variables related to the genes that have
mutations (`1` if the gene have mutation for that individual, `0`
otherwise) and all the variable that showed a significant association in
the analysis above. In order to perform an additional variable selection
I add a random variable to my dataset. Then checking the variable
importance I will exclude all the variable that will have an importance
lower than my random variable:

``` r
DT$random_variable = rnorm(NROW(DT),0,1)
```

I have a low sample size considering the missing data, not enough to
split between **train** and **test** dataset. *Random Forest algorithm*
(by it’s implementation) can handle that problems using the **out-of-bag
error**. The out-of-bag (**OOB**) error is the average error for each
calculated using predictions from the trees that do not contain in their
respective bootstrap sample, in this way I can avoid the test-set and
also the cross-validation. Now I run an optimization grid to look for
the parameters that minimize the OOB error:

``` r
rf_grid = data.table(expand.grid(mtry = 5:40,
                                 ntree = seq(1000,3000,250),
                                 nodesize = 1:10))
rf_grid$ID = 1:NROW(rf_grid)

n_cores = detectCores() - 2
cl = makeCluster(n_cores)
clusterEvalQ(cl, {
  library(randomForest)
  library(data.table)
  library(stringr)
})

clusterExport(cl, varlist = c("DT","rf_grid"))

rf_opt = parLapplyLB(cl,1:NROW(rf_grid), function(j){
  a = Sys.time()
  set.seed(j)
  rf = randomForest(as.factor(Outcome_num) ~ .,
                    data = DT[,-c("ID"),with=F],
                    na.action = "na.omit",
                    mtry = rf_grid$mtry[j],
                    replace = F,
                    nodesize = rf_grid$nodesize[j],
                    ntree = rf_grid$ntree[j])
  b = Sys.time()
  cat(paste0("Grid ",str_pad(j, 4, pad = "0"),
             " - ",str_pad(round(as.numeric(b-a),4),6,side = "right", pad = "0"),
             " secs - ntree: ",rf_grid$ntree[j]," - mtry: ",
             str_pad(rf_grid$mtry[j], 2, pad = "0"),
             " - nodesize: ",str_pad(rf_grid$nodesize[j], 2, pad = "0"),
             " - oob_err_rate: ",str_pad(round(rf$err.rate[rf_grid$ntree[j],1],4),6,
                                         side = "right", pad = "0"),
             "\n"),
      file = "rf_model.log",
      append = TRUE)
  return(NULL)
})
```

Checking the results from the file `rf_model.log`. Here the parameters
that produced the best performance:

``` r
rf_opt = fread("rf_model.log")
rf_opt = rf_opt[,c(2,4,8,11,14,17)]
setnames(rf_opt,names(rf_opt),c("ID","secs","ntree","mtry","nodesize","err_rate_oob"))
rf_opt[order(err_rate_oob)][1]
```

    ##     ID   secs ntree mtry nodesize err_rate_oob
    ## 1: 407 0.3092  1500   15        2       0.1383

Let’s explore the variable importance grid:

``` r
set.seed(407)
rf_model = randomForest(as.factor(Outcome_num) ~ .,
                  data = DT[,-c("ID"),with=F],
                  na.action = "na.omit",
                  mtry = 15,
                  replace = F,
                  nodesize = 2,
                  ntree = 1500)
vimp = data.table(variables = rownames(varImp(rf_model)), varImp(rf_model))
vimp[order(vimp$Overall,decreasing = TRUE)]
```

    ##           variables     Overall
    ##  1:      Tumor.Size 3.282095879
    ##  2:    Primary.Site 3.069293877
    ##  3:           Stage 2.260759411
    ##  4:           Grade 1.872061366
    ##  5:               T 1.606236128
    ##  6:             Age 1.450146633
    ##  7: random_variable 1.290050410
    ##  8:           PTCH1 0.602778771
    ##  9:       KRAS_Col1 0.436133561
    ## 10:          NOTCH1 0.415502223
    ## 11:           STK11 0.378055681
    ## 12:        ATM_Col1 0.373656253
    ## 13:          CDKN2A 0.370080421
    ## 14:       Histology 0.363434677
    ## 15:               N 0.335763613
    ## 16:       ratio_mut 0.315819728
    ## 17:            MSH2 0.301815605
    ## 18:          DNMT3A 0.260169503
    ## 19:            EGFR 0.247657551
    ## 20:          PDGFRB 0.242806433
    ## 21:          PIK3CB 0.194797421
    ## 22:       TP53_Col1 0.169450757
    ## 23:           HNF1A 0.164890766
    ## 24:       Radiation 0.157968784
    ## 25:            AKT1 0.148541470
    ## 26:               M 0.147860513
    ## 27:           ERBB4 0.125689819
    ## 28:          MAP2K2 0.123720936
    ## 29:          PIK3CA 0.120833753
    ## 30:           FGFR3 0.119550755
    ## 31:        ALK_Col2 0.108852249
    ## 32:           FBXW7 0.107002464
    ## 33:         SMARCB1 0.104346088
    ## 34:         SMARCA4 0.101209231
    ## 35:          CTNNB1 0.096547633
    ## 36:            MSH6 0.096378289
    ## 37:        ALK_Col1 0.095983657
    ## 38:        MLH_Col2 0.089480102
    ## 39:       POLD_Col2 0.073695680
    ## 40:            TERT 0.070177069
    ## 41:           FOXL2 0.064526350
    ## 42:           ERBB3 0.061559707
    ## 43:           FGFR1 0.055112386
    ## 44:         NF_Col2 0.053687603
    ## 45:         NF_Col5 0.042505586
    ## 46:           NTRK1 0.039510291
    ## 47:         NF_Col1 0.039066759
    ## 48:             SMO 0.038960649
    ## 49:         NF_Col3 0.037808241
    ## 50:       TP53_Col2 0.034205497
    ## 51:            GNAS 0.028843853
    ## 52:             MET 0.024266902
    ## 53:            TSC2 0.020100494
    ## 54:           CCND2 0.019488531
    ## 55:            PTEN 0.017718135
    ## 56:             RB1 0.015918276
    ## 57:            BRAF 0.008024137
    ## 58:             APC 0.000000000
    ## 59:        ATM_Col2 0.000000000
    ## 60:            ESR1 0.000000000
    ## 61:            FLT4 0.000000000
    ## 62:       KRAS_Col2 0.000000000
    ##           variables     Overall

In the table I can notice that only 5 variables are above my random
variable. So let’s try to rerun the model considering only
`Tumor.Size`,`Primary.Site`,`Stage`,`Grade`,`T`. I create a new
optimization grid to have the best parameters for the new model:

``` r
rf_grid_2 = data.table(expand.grid(mtry = 1:5,
                                   ntree = seq(1000,3000,250),
                                   nodesize = 1:10))
rf_grid_2$ID = 1:NROW(rf_grid_2)

n_cores = detectCores() - 2
cl = makeCluster(n_cores)
clusterEvalQ(cl, {
  library(randomForest)
  library(data.table)
  library(stringr)
})

clusterExport(cl, varlist = c("DT","rf_grid_2"))

rf_opt = parLapplyLB(cl,1:NROW(rf_grid_2), function(j){
  a = Sys.time()
  set.seed(j)
  rf = randomForest(as.factor(Outcome_num) ~ Stage + Primary.Site + Tumor.Size + `T` + Grade,
                    data = DT[,-c("ID"),with=F],
                    na.action = "na.omit",
                    mtry = rf_grid_2$mtry[j],
                    replace = F,
                    nodesize = rf_grid_2$nodesize[j],
                    ntree = rf_grid_2$ntree[j])
  b = Sys.time()
  cat(paste0("Grid ",str_pad(j, 4, pad = "0"),
             " - ",str_pad(round(as.numeric(b-a),4),6,side = "right", pad = "0"),
             " secs - ntree: ",rf_grid_2$ntree[j]," - mtry: ",
             str_pad(rf_grid_2$mtry[j], 2, pad = "0"),
             " - nodesize: ",str_pad(rf_grid_2$nodesize[j], 2, pad = "0"),
             " - oob_err_rate: ",str_pad(round(rf$err.rate[rf_grid_2$ntree[j],1],4),6,
                                         side = "right", pad = "0"),
             "\n"),
      file = "rf_model_2.log",
      append = TRUE)
  return(NULL)
})
```

Load the file with the optimization grid results from file
`rf_model_2.log`:

``` r
rf_opt = fread("rf_model_2.log")
rf_opt = rf_opt[,c(2,4,8,11,14,17)]
setnames(rf_opt,names(rf_opt),c("ID","secs","ntree","mtry","nodesize","err_rate_oob"))
rf_opt[order(err_rate_oob)][1]
```

    ##    ID   secs ntree mtry nodesize err_rate_oob
    ## 1: 48 0.2718  1000    3        2       0.0714

Analysis of the final model:

``` r
set.seed(48)
rf_model = randomForest(as.factor(Outcome_num) ~ Stage + Primary.Site + 
                          Tumor.Size + `T` + Grade,
                  data = DT[,-c("ID"),with=F],
                  na.action = "na.omit",
                  mtry = 3,
                  replace = F,
                  nodesize = 2,
                  ntree = 1000)
rf_model
```

    ## 
    ## Call:
    ##  randomForest(formula = as.factor(Outcome_num) ~ Stage + Primary.Site +      Tumor.Size + T + Grade, data = DT[, -c("ID"), with = F],      mtry = 3, replace = F, nodesize = 2, ntree = 1000, na.action = "na.omit") 
    ##                Type of random forest: classification
    ##                      Number of trees: 1000
    ## No. of variables tried at each split: 3
    ## 
    ##         OOB estimate of  error rate: 7.14%
    ## Confusion matrix:
    ##    0  1 class.error
    ## 0 19  6  0.24000000
    ## 1  1 72  0.01369863

From the summary results we can see that the OOB estimate of error rate
is 7.14%. The model makes 24% of 1<sup>st</sup> type error and 1.3%
2<sup>nd</sup> type error. The performance for the 1<sup>st</sup> type
error is not good but it gives a more conservative prediction.

Having more time and resources I could make a better merge between the
classes, try to impute the variables that contains missing data and use
the correlation/association between the variables. Having a better
knowledge of the lungs I could also make assumption and/or provide a
cleaner dataset for the analysis. Working with `lightgboost` or
`xgboost` could provide a more refined estimations.
