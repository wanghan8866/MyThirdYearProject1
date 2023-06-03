# syntax=docker/dockerfile:1
FROM python:3.9-alpine

RUN apk add --no-cache --update \
    python3 python3-dev gcc \
    gfortran musl-dev
 
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install flappy_bird_gym
RUN pip install -r requirements.txt

COPY . .
CMD ["python3", "main.py"]
