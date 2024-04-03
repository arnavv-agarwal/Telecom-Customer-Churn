## Overview
This project focuses on investigating opportunities to decrease customer churn at S-mobile, a mobile service provider. The dataset includes three parts: a training sample with a 50% churn rate, a test sample with a 50% churn rate, and a representative sample with a churn rate of 2%, reflecting the actual monthly churn rate for S-mobile.

## Dataset
- Training Sample: 27,300 observations with a 50% churn rate.
- Test Sample: 11,700 observations with a 50% churn rate.
- Representative Sample: 30,000 observations with a churn rate of 2%.

```python 
## load the data - this dataset must NOT be changed
s_mobile = pd.read_pickle("data/s_mobile.pkl")
s_mobile["churn_yes"] = rsm.ifelse(s_mobile["churn"] == "yes", 1, 0)
```

## Analysis
Using a logit model, the likelihood of customer churn risk was predicted. The relative importance of features was assessed using the odd_ratio function, highlighting key variables influencing churn risk.

```python 
lr2 = smf.glm(
    formula="churn_yes ~ changer + changem + mou + \
    overage + months + uniqsubs +  \
    retcalls + dropvce + eqpdays + refurb + \
    highcreditr + mcycle + \
    travel + region + occupation + \
    churn:changer + churn:changem  + occupation:mou + \
    occupation:months + months:retcalls + \
    retcalls:churn + months:churn + churn:months + \
    churn:overage + overage:region",    family=Binomial(link=logit()),
    freq_weights=s_mobile_train.loc[s_mobile_train.training == 1, "cweight"],
    data= s_mobile_train.query("training == 1")
).fit(cov_type="HC1")
lr2.summary()
```

### Important Features
- 'occupation[T.student]': Students are 1.845 times more likely to churn.
- 'occupation[T.professional]': Professionals are 1.451 times more likely to churn.
- 'eqpdays': Customers using the current handset for fewer days are 1.370 times more likely to churn.
- 'overage': Customers with higher overage are 3.607 times more likely to churn.
- 'refurb[T.yes]': Customers with refurbished smartphones are 1.325 times more likely to churn.
- 'retcalls': Customers with more calls to the retention team are 1.549 times more likely to churn.

## Action Plans
1. **Student Discounts**: Offer 15% monthly discount to students to reduce churn rate.
2. **Promotional Smartphone**: Provide new handsets to customers nearing contract end to retain them.
3. **Target Retired Professionals**: Offer family packs with 20% discount to retain retired customers.
4. **Manage Overage**: Upgrade overage customers to premium plan with 2 months free subscription.

## Assumptions
```python 
# list your assumptions here
monthly_revenue = 30
annual_growth = 0.03
annual_discount_rate = 0.1
monthly_discount_rate = (1+annual_discount_rate)**(1/12)-1
cost_service = 0.15*monthly_revenue
marketing_cost = 0.05*monthly_revenue
nr_years = 5
```


### Under Representative Sample:
1. **Student Discount**: Target students based on feature importance.
2. **Promotional Smartphone**: Target customers based on 'eqpdays' and 'refurb'.

## Customer Lifetime Value (CLV)
- Implementing the free handset incentive increases CLV by approximately $146 compared to no strategy.
- Implementing family packs increases CLV by approximately $460 compared to no strategy.

Through these strategies, we aim to reduce churn rate and increase customer lifetime value.

