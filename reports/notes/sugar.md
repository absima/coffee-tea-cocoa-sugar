### Market Behaviour

* The sugar market exhibits volatile monthly log returns between **2019-07-01** and **2026-01-01**, with a total of **79 rows in the test period**.
* During this period, the market experienced both positive and negative returns, with the highest absolute error occurring on **2025-05-01** at 0.0670940996215047.
* The most recent data points indicate a continued volatile trend, with high absolute errors observed for the months of **May, June**, and **July 2025**.

### Model Performance

* The Ridge regression model (alpha=1.0) achieved a mean absolute error (MAE) of **0.05047930573100224** across the test period.
* The model's root mean squared error (RMSE) was **0.06795241748795643**, indicating that it tends to overestimate large errors.
* Correlation between true and predicted values was relatively low at -0.06845269064687395, suggesting a need for further refinement.

### Confidence & Risks

* The model's performance during the recent window (12 months) shows a mean absolute error of **0.03366799170974332**, which is higher than the overall MAE.
* The highest absolute error in this period occurred on **2025-05-01** at 0.0670940996215047, indicating potential for model improvement.
* Continued monitoring of the market and refinement of the model are necessary to ensure its accuracy and reliability.
