# Task 5 - Analysis

## 5.1 Set A Normal Equation Weights

The Normal Equation coefficients for Set A were:

- Intercept (theta0): 6890.57
- Irradiation: 8345.04
- Module temperature: -108.12
- Ambient temperature: -17.16
- sin(hour): -47.38
- cos(hour): -416.56

The feature with the largest coefficient magnitude, excluding the
intercept, was irradiation with a coefficient of 8345.04.

The irradiation coefficient is positive. This makes physical sense
because higher solar irradiation generally means that more solar energy
is available to the PV modules, which normally results in higher AC
power generation.

The temperature coefficients were negative in this fitted model. This
indicates that after considering the other features, increasing
temperature was associated with a small decrease in predicted power.


## 5.2 Set A vs Set B Daytime Accuracy

The daytime RMSE obtained using the Normal Equation was:

- Set A daytime RMSE: 704.46
- Set B daytime RMSE: 3409.50

The difference between the two models was:

3409.50 - 704.46 = 2705.04

The peak hourly AC power in the dataset was:

27325.90

Therefore, the RMSE difference as a percentage of peak hourly power was:

(2705.04 / 27325.90) * 100 = 9.90%

Set B had a much larger daytime RMSE than Set A.

Set A uses measurements collected directly at the solar plant,
including irradiation, module temperature and ambient temperature.
Set B uses public weather information such as shortwave radiation,
temperature at 2 m and cloud cover.

For this dataset, the on-site sensor measurements were more suitable
for accurate power prediction. Public weather data can still be useful
when local sensors are unavailable, but the higher RMSE shows that it
produced less accurate predictions for this plant.


## 5.3 Comparison of Regression Solvers

Three regression methods were implemented:

1. Normal Equation
2. Batch Gradient Descent
3. Stochastic Gradient Descent

The Set A Normal Equation coefficients were:

[6890.56596, 8345.04388, -108.12084, -17.16187,
 -47.37639, -416.55781]

The final Set A Batch Gradient Descent coefficients were:

[6890.56596, 8345.04377, -108.12069, -17.16192,
 -47.37640, -416.55781]

The Set A SGD coefficients were:

[6892.53926, 5892.62470, 3116.33049, -973.44482,
 -48.64218, -412.19517]

The maximum absolute difference between the final Batch Gradient
Descent and Normal Equation coefficients for Set A was approximately:

0.000146

This very small difference shows that the final Batch Gradient Descent
solution converged very closely to the Normal Equation solution.

The required Batch GD learning-rate experiment was performed for
500 iterations. Among the tested learning rates, 0.0001 was stable
and gave a much lower final cost than 0.00001, while 0.001 diverged.
A longer final Batch GD run was then used to verify convergence with
the Normal Equation.

The SGD learning-rate experiment was performed for 50 epochs.
Among the tested SGD learning rates, 0.01 produced the lowest final
training cost.

For this relatively small dataset, I would prefer the Normal Equation
because it directly calculates the solution and does not require
learning-rate tuning.

For a dataset containing around 10 million rows, I would prefer an
iterative optimization method such as SGD because directly forming
and solving the Normal Equation becomes less attractive for very large
datasets. SGD processes observations incrementally and can scale better
to large amounts of data.

### Optimization Decision Matrix

| Method | Learning Rate Required | Iterative | Suitable for This Dataset | Suitable for Very Large Dataset |
|---|---|---|---|---|
| Normal Equation | No | No | Yes | Less suitable |
| Batch GD | Yes | Yes | Yes | Possible but each update uses all rows |
| SGD | Yes | Yes | Yes | More suitable |


## 5.4 Batch GD vs SGD Cost Curves

The Batch Gradient Descent learning-rate graph showed very different
behavior for the three tested learning rates.

For Batch GD, learning rates 0.00001 and 0.0001 remained stable.
The learning rate 0.001 caused the cost to increase to an extremely
large value, approximately 2.93 x 10^101 after 500 iterations.
This indicates divergence because the learning rate was too large.

The final costs from the Batch GD learning-rate experiment were:

- alpha = 0.00001: 588777442.31
- alpha = 0.0001: 121943628.68
- alpha = 0.001: 2.93 x 10^101

Therefore, 0.0001 was selected for the final Batch GD model.

For SGD, the final costs after 50 epochs were:

- alpha = 0.0001: 590814652.69
- alpha = 0.001: 122144480.26
- alpha = 0.01: 95970051.93

The SGD graph shows that the larger tested learning rate reached a
lower cost more quickly.

Batch Gradient Descent calculates each update using the complete
training dataset, so its optimization path is generally smoother.
SGD updates the parameters after individual training examples, so its
optimization path can fluctuate more because every update is based on
one observation instead of the complete training dataset.


## 5.5 Residual Analysis

Residual was calculated as:

residual = actual AC power - predicted AC power

The residual plot for the Set A Normal Equation showed that the errors
were relatively small during night-time hours when solar generation was
close to zero.

The largest spread in residuals occurred mainly during daylight and
high-generation hours, particularly around approximately 10:00 to
15:00. A large positive residual can be observed around 11:00, while
a large negative residual can be observed around 14:00.

A positive residual means that the actual power was higher than the
predicted power. A negative residual means that the model predicted
more power than was actually generated.

One possible physical reason for the larger daytime errors is that
solar power does not depend only on the variables included in the
linear regression model. Changes in clouds, module conditions,
temperature, inverter operation and rapidly changing irradiation can
cause actual power to differ from a simple linear prediction.

Overall, the residual plot shows that prediction errors are more
significant during the main solar-generation period than during
night-time hours.