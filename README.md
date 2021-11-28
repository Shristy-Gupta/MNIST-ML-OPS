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

we can now access these apis from the links http://localhost:5000/decision_tree_predict and http://localhost:5000/svm_predict
to see the oupout we can either give curl command or simply check from postman

If we are giving curl commands it will be as follows:
``` sh
curl http://localhost:5000/svm_predict -X POST  -H 'Content-Type: application/json' -d '{"image":
["0.0","0.0","0.0","2.000000000000008","12.99999999999999","2.3092638912203262e-14","0.0","0.0","0.0","0.0","0.0","7.99999999999998","14.999999999999988","2.664535259100375e-14","0.0","0.0","0.0","0.0","4.9999999999999885","15.999999999999975","5.000000000000027","2.0000000000000027","3.552713678800496e-15","0.0","0.0","0.0","14.999999999999975","12.000000000000007","1.0000000000000182","15.999999999999961","4.000000000000018","7.1054273576009955e-15","3.5527136788004978e-15","3.9999999999999925","15.999999999999984","2.0000000000000275","8.999999999999984","15.999999999999988","8.00000000000001","1.4210854715201997e-14","3.1554436208840472e-30","3.5527136788004974e-15","9.999999999999995","13.999999999999986","15.99999999999999","16.0","4.000000000000025","7.105427357601008e-15","0.0","0.0","0.0","0.0","12.999999999999982","8.000000000000009","1.4210854715202004e-14","0.0","0.0","0.0","0.0","0.0","12.999999999999982","6.000000000000012","1.0658141036401503e-14","0.0"]}'
```

``` sh
curl http://localhost:5000/decision_tree_predict -X POST  -H 'Content-Type: application/json' -d '{"image":
["0.0","0.0","0.0","2.000000000000008","12.99999999999999","2.3092638912203262e-14","0.0","0.0","0.0","0.0","0.0","7.99999999999998","14.999999999999988","2.664535259100375e-14","0.0","0.0","0.0","0.0","4.9999999999999885","15.999999999999975","5.000000000000027","2.0000000000000027","3.552713678800496e-15","0.0","0.0","0.0","14.999999999999975","12.000000000000007","1.0000000000000182","15.999999999999961","4.000000000000018","7.1054273576009955e-15","3.5527136788004978e-15","3.9999999999999925","15.999999999999984","2.0000000000000275","8.999999999999984","15.999999999999988","8.00000000000001","1.4210854715201997e-14","3.1554436208840472e-30","3.5527136788004974e-15","9.999999999999995","13.999999999999986","15.99999999999999","16.0","4.000000000000025","7.105427357601008e-15","0.0","0.0","0.0","0.0","12.999999999999982","8.000000000000009","1.4210854715202004e-14","0.0","0.0","0.0","0.0","0.0","12.999999999999982","6.000000000000012","1.0658141036401503e-14","0.0"]}'
```




