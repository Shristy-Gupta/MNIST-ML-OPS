
### Assignment 6
![Assignment5 result screenshot](https://user-images.githubusercontent.com/26459890/136364943-be734619-10a1-4c4c-a37e-c19d43b00ace.PNG)
Both the test cases are executed

1) Tests re present in test/test_utils.py
2) First function is test_model_writing: This function internally tests the run_classification_experiment(). This function is present in utility.py. Basically this function checks if the result is coming and the result is present in the model folder as joblib file or not.
![image](https://user-images.githubusercontent.com/26459890/136365756-a9938ac0-53c7-471b-b6cb-5265f8d4c70e.png)

4) Second function is test_small_data_overfit_checking: This function  checks  the accuracy and f1 score at a certain threshold. I have chosen threshold as 0.8 for both the parameters. Also I have taken only 50% of the dataset for checking the overfitting.
 ![image](https://user-images.githubusercontent.com/26459890/136366043-d21854a3-5b6b-49b6-b9ee-ab9cf1815208.png)






