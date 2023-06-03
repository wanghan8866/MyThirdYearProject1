# syntax=docker/dockerfile:1
FROM python:3.9-alpine

COPY requirements.txt requirements.txt
RUN pip install flappy_bird_gym
RUN pip install -r requirements.txt

COPY . .
CMD ["python3", "main.py"]