# ML Interview Coding Challenge

> [!NOTE]
> Useful information how to start the docker application

Welcome to the repository of the HUK Coding Challenge for ML Engineers. 


to run the docker file to test the API please do the following: 
- navigate to the folder docker_app/app and run in terminal 
```
docker built -t "name_env" . 
docker run -it --name "container_name" -p 8080:8080 "name_env"
```

- in a seperate terminal you can now send POST requests to the server with 

```
curl -X POST -H "Content-Type: multipart/form-data" http://localhost:8080/infer -F "file=@image.png"
```

and enjoy the prediction on a real-world random image of a car


> [!IMPORTANT]
> Server is not production ready, only for demonstration purposes. For further processing use WSGI server instead.

- Next steps could be using docker-compose witg nginx reverse-proxy and uwsgi server


