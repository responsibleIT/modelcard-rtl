FROM python:3.11

COPY ./ /app

WORKDIR  /app

RUN pip3 install -r requirements.txt

CMD ["python3", "main.py"]