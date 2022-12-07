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
