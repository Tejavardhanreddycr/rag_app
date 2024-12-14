FROM python:3.10.12

WORKDIR /app 

COPY backend/ . 

RUN pip install -r requirements.txt

EXPOSE 8088

ENTRYPOINT [ "python", "app.py" ]
