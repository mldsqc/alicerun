FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y wget

RUN apt-get install -y python3 python3-pip
COPY . .
RUN pip3 install -r requirements.txt
RUN pip3 install O365

CMD python3 /src/Telegram_bot/bot.py
#CMD python3 /src/Telegram_bot/microsoft_todo_import_3.py
#CMD python3 /src/Telegram_bot/testing_playground.py


#CMD python3 /src/Telegram_bot/make_postgres_db.py
#RUN make /app

#ENTRYPOINT [ "python", "bot.py" ]

#RUN /usr/local/bin/python -m pip install --upgrade pip


#WORKDIR /app
