# README
A very (very) basic FastAPI 

## Quickstart

Either run natively:

```
$ pip install -r requirements.txt
$ uvicorn "app.simple:app" --host 0.0.0.0 --port 5000
```

Or run using Docker:
```
$ docker build -t fastapi_demo .
$ docker run -d --name fastapi_demo -p 5000:5000 fastapi_demo
```
# LICENSING
* `imagenet_classes.json`: Taken from [raghakot/keras-vis](https://github.com/raghakot/keras-vis), under MIT license.