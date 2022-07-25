### What is it?

This repository including notebooks with examples about timeseries anomaly detection.

YouTube course https://www.youtube.com/watch?v=92EF4vqaBSE&list=PL7GGfr9mTeYWniRK11xuFsEky07oUQ_tX&index=2

Our dataset is timeseries of food retail:

| ou           | datetime            | cheques | rto   | n_sku | cnt | cashnum |
|--------------|---------------------|---------|-------|-------|-----|---------|
| 468          | 2019-11-16 08:00:00 | 34      | 8003  | 137   | 173 | 3       |
| 468          | 2019-11-16 09:00:00 | 40      | 20129 | 283   | 517 | 2       |

* ou - index of shop
* datetime - ISO format of date and hour
* cheques - count of payment
* rto - revenue in rubles
* n_sku - count of lines in bills
* cnt - number of items
* cashnum - number of opened windows while hour

We explore few approaches for anomaly detection in 1-D timeseries such as:
1. statistical anomalies based on normal distribution
2. forecasting method - detection anomalies as error of forecast
3. classification method Isolation Forest
4. clusterization method K-means
5. K Nearest Neighbors

Also we explore few approaches for anomaly detection in Multi dimensions 
timeseries based on PyOD library.