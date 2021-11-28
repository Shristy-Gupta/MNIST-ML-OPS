# Course ML-OPS
### Assessment 10: API Creation and Docker Deployment
### Submitted By: Shristy Gupta M20CS015
### Submitted To: Dr. Tejas Indulal Dhamecha


### API Creation in FLASK and POSTMAN for seeing results
Two APIs created svm_predict and decision_tree_predict with method type POST
Functionality: The apis take json Body which contains the attribute Image and outputs the predicted Number
![image](https://user-images.githubusercontent.com/26459890/143763613-7e75245d-9226-4277-adc8-4844e2e0e9ca.png)
![image](https://user-images.githubusercontent.com/26459890/143763621-96e0aad9-ab77-4305-b649-c6ba84b9ea42.png)

To host these apis :
``` sh
e:; cd '<path>'; & 'D:\Anaconda\python.exe' '<path>\extensions\ms-python.python-2021.11.1422169775\pythonFiles\lib\python\debugpy\launcher' '62702' '--' '-m' 'flask' 'run' '--no-debugger'
```
After this our apis will be hosted and the status of the terminal will be as follows:
![image](https://user-images.githubusercontent.com/26459890/143763726-16affda2-603e-4d2d-bdf8-32304450d79e.png)


