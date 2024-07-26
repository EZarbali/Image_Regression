# HUK_interview

> [!NOTE]
> Useful information how to start the docker application

Welcome to the repository of the HUK Coding Challenge for ML Engineers. 


to run the docker file to test the API please do the following: 
- go to the folder docker_app/app
```
docker built -t "name_env" . 
docker run -it --name "container_name" -p 8080:8080 "name_env"
```

- in a seperate terminal you can now send POST requests to the server


