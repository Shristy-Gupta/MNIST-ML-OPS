
### Final Exam

Question 1 solution
![image](https://user-images.githubusercontent.com/26459890/143684743-c634a36c-c254-40ab-85e1-de31a6ca6be5.png)

1) Do you see any variations amoung these runs? Why and why not?
I could not see any variations among any of these runs. One reason can be that the train test dev split choosen in each runs were almost same

2) Identify and Report the Bad hyperparameter that is leading to over-fitting
Among the given gamma values : gamma = [1e-7, 1e-5, 1e-3, 0.01, 0.02, 0.1] gamma 1e-7 performed the worst leading to overfitting.
In the screenshot above the value against underperforming gamma is coming as 0.0 but actually its 1e-7

3) Identify what is a good hyperparameter
The best performing gamma is 1 which is approximately 0.95

4) Comment on Interpretetaions on Hyper-paramters, values and performance
As the best Hyper-paramters is choosen then the corresponding accuracy also improves. There is no sign of increase of Hyper-parameter and increase of performance.


