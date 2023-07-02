FROM python:3.11.3-bullseye

COPY requirements.txt requirements.txt 

RUN pip3 install -r requirements.txt

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . .

CMD python src/app.py