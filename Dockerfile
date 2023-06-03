# syntax=docker/dockerfile:1
FROM python:slim
WORKDIR /app_home

RUN apk add --no-cache --update \
    python3 python3-dev gcc \
    gfortran musl-dev \
    sdl2-dev
 
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install flappy_bird_gym
RUN pip install -r requirements.txt

COPY . .
CMD ["python3", "main.py"]
