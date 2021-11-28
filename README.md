# Course ML-OPS
### Assessment 10: API Creation and Docker Deployment
### Submitted By: Shristy Gupta M20CS015
### Submitted To: Dr. Tejas Indulal Dhamecha


# PART 1 API Creation in FLASK and POSTMAN for seeing results
1) Two APIs created svm_predict and decision_tree_predict with method type POST
2) Functionality: The apis take json Body which contains the attribute Image and outputs the predicted Number

![image](https://user-images.githubusercontent.com/26459890/143763613-7e75245d-9226-4277-adc8-4844e2e0e9ca.png)
![image](https://user-images.githubusercontent.com/26459890/143763621-96e0aad9-ab77-4305-b649-c6ba84b9ea42.png)

3) To host these apis :
``` sh
e:; cd '<path>'; & 'D:\Anaconda\python.exe' '<path>\extensions\ms-python.python-2021.11.1422169775\pythonFiles\lib\python\debugpy\launcher' '62702' '--' '-m' 'flask' 'run' '--no-debugger'
```
4) After this our apis will be hosted and the status of the terminal will be as follows:
![image](https://user-images.githubusercontent.com/26459890/143763726-16affda2-603e-4d2d-bdf8-32304450d79e.png)

5) we can now access these apis from the links http://localhost:5000/decision_tree_predict and http://localhost:5000/svm_predict
to see the oupout we can either give curl command or simply check from postman

6) If we are giving curl commands it will be as follows:
``` sh
curl http://localhost:5000/svm_predict -X POST  -H 'Content-Type: application/json' -d '{"image":
["0.0","0.0","0.0","2.000000000000008","12.99999999999999","2.3092638912203262e-14","0.0","0.0","0.0","0.0","0.0","7.99999999999998","14.999999999999988","2.664535259100375e-14","0.0","0.0","0.0","0.0","4.9999999999999885","15.999999999999975","5.000000000000027","2.0000000000000027","3.552713678800496e-15","0.0","0.0","0.0","14.999999999999975","12.000000000000007","1.0000000000000182","15.999999999999961","4.000000000000018","7.1054273576009955e-15","3.5527136788004978e-15","3.9999999999999925","15.999999999999984","2.0000000000000275","8.999999999999984","15.999999999999988","8.00000000000001","1.4210854715201997e-14","3.1554436208840472e-30","3.5527136788004974e-15","9.999999999999995","13.999999999999986","15.99999999999999","16.0","4.000000000000025","7.105427357601008e-15","0.0","0.0","0.0","0.0","12.999999999999982","8.000000000000009","1.4210854715202004e-14","0.0","0.0","0.0","0.0","0.0","12.999999999999982","6.000000000000012","1.0658141036401503e-14","0.0"]}'
```

``` sh
curl http://localhost:5000/decision_tree_predict -X POST  -H 'Content-Type: application/json' -d '{"image":
["0.0","0.0","0.0","2.000000000000008","12.99999999999999","2.3092638912203262e-14","0.0","0.0","0.0","0.0","0.0","7.99999999999998","14.999999999999988","2.664535259100375e-14","0.0","0.0","0.0","0.0","4.9999999999999885","15.999999999999975","5.000000000000027","2.0000000000000027","3.552713678800496e-15","0.0","0.0","0.0","14.999999999999975","12.000000000000007","1.0000000000000182","15.999999999999961","4.000000000000018","7.1054273576009955e-15","3.5527136788004978e-15","3.9999999999999925","15.999999999999984","2.0000000000000275","8.999999999999984","15.999999999999988","8.00000000000001","1.4210854715201997e-14","3.1554436208840472e-30","3.5527136788004974e-15","9.999999999999995","13.999999999999986","15.99999999999999","16.0","4.000000000000025","7.105427357601008e-15","0.0","0.0","0.0","0.0","12.999999999999982","8.000000000000009","1.4210854715202004e-14","0.0","0.0","0.0","0.0","0.0","12.999999999999982","6.000000000000012","1.0658141036401503e-14","0.0"]}'
```
7) To check the Results from Postman
use post command with urls http://localhost:5000/decision_tree_predict and http://localhost:5000/svm_predict and give above body in JSON format
![image](https://user-images.githubusercontent.com/26459890/143763961-8d25787a-0aaa-4f05-96b5-387f7777ee67.png)

![image](https://user-images.githubusercontent.com/26459890/143764007-6e415c3e-7976-44f7-8745-de70beba14e4.png)


# Part 2 Dockerising the deployment
1) Installed the docker for windows from docker toolbox, since the virtualbox and docker has conflicting configurations (Hyper-V conflict) from https://docs.docker.com/desktop/windows/install/ 
2) 2) Created account docker hub with username shristy26 and a new repository shristy26/m20cs015server
``` sh
FROM ubuntu
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install apache2 -y
RUN apt-get install apache2-utils -y
RUN apt-get clean
EXPOSE 80
CMD ["apache2ctl","-D","FOREGROUND"]
```
3)![image](https://user-images.githubusercontent.com/26459890/131547012-d4eeb469-cf38-44ed-99db-779f67a2db90.png)
3) Created a dockerfile in m20cs015server which creates ubuntu as base image. On top of the base image installed apache2 for creating a new docker image from scratch. The code for the same is available in m20cs015server folder in "Dockerfile" 
4) In order to pull the image in sudokuwebapp (in order to deploy it) pushed the image to docker hub webserver 
``` sh
docker build . -t m20cs015server
docker image build -t shristy26/m20cs015server:ver1 .
docker image push shristy26/m20cs015server:ver1
```
After successful execution I could see the tag in docker hub as follows with Digest as sha256:a4c5a1a49d3b696898ecd75263de99fb1afefd7509a147b1ad33de7d4957280e
![image](https://user-images.githubusercontent.com/26459890/131549740-d9318bf8-2980-43f8-adf6-24d0c7aab6a7.png)
5) After I pulled the container with python and flask images installed

``` sh
FROM python:3.8.8
ADD . /prediction
WORKDIR /prediction
RUN pip install -r requirements.txt
``` 
6) To build docker image of the webapp :
``` sh
docker build .  -t m20cs015
```
7) To run this image in specific port used following command
``` sh
docker run -it -p 5000:5000 m20cs015
```
8) Finally the app will run on url: http://localhost:5000/prediciton/

#### Steps to run the file
1) First we need to run Docker Compose
``` sh
docker-compose up. This file has **ports where the apis will be hosted and will provide some disk space to the program if needed**
```
2) Now when docker compose is up and running we need to move to CLI command and run the bash file. This file has **Curl Commands**
``` sh
Docker exec -it mlops /bin/bash
bash docker_example.sh
```
3) Now we can access the apis





