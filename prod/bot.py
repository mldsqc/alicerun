# -*- coding: utf-8 -*-
# Import Library
# from mysql.connector import MySQLConnection, Error
# from python_mysql_dbconfig import read_db_config
from telebot import types
from datetime import date
from datetime import datetime
from python_postgresql_dbconfig import read_db_config

from pathlib import Path
import psycopg2
import telegram
import telebot
import pandas as pd
import os.path
import configparser
import today_box

# path = Path(__file__)
# ROOT_DIR = path.parent.absolute()
# config_path = os.path.join(ROOT_DIR, "config.ini")
import configparser

config = configparser.ConfigParser()
config.read('/src/Telegram_bot/config.ini')
# config.read(config_path)

bot = telebot.TeleBot(config.get('telegram-bot', 'TOKEN'))
print("\nToday's date:", date.today(), " Working on you my Lord!")

# init user-employee
@bot.message_handler(commands=["start"])
def start(m, res=False):
    bot.send_message(m.chat.id,
                     '✌️ *Привіт* \n Проверяемся, еблан, на эмоциональную стойкость  \n\n'
                     '\n\n 🔻 /today 🔻',
                     parse_mode=telegram.ParseMode.MARKDOWN)

# send daily questions
@bot.message_handler(commands=["today"])
def today(m, res=False):
    today_box.today_send(m.chat.id)


# SAVE ANSWER
@bot.callback_query_handler(lambda call: True)
def handle(call):
    chat_id = call.message.chat.id
    date_answer = date.today()
    time_answer = datetime.now().strftime("%H:%M:%S")
    username = call.message.chat.username
    first_name = call.message.chat.first_name
    last_name = call.message.chat.last_name
    location = "None"

    event = str(call.message.text)
    event_description = call.data


    # """ Connect to MySQL database """
    # db_config = read_db_config()
    # conn = None
    # try:
    #     conn = MySQLConnection(**db_config)
    #     cursor=conn.cursor()
    #
    #     sql = "INSERT INTO `daily_answers` (`chat_id`, `date_answer`, `time_answer`, `username`, `first_name`, `last_name`, `location`, `event`, `event_description`) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    #     cursor.execute(sql, (str(chat_id), str(date_answer), str(time_answer), str(username), str(first_name), str(last_name), str(location), str(event), str(event_description)))
    #
    #     conn.commit()
    # except Error as error:
    #     print("Answer not added: ", str(chat_id), " ", str(date_answer), " ", str(time_answer), " ", str(username), " ", str(first_name), " ", str(last_name), " ", str(location), " ", str(event), " ", str(event_description))
    # finally:
    #     if conn is not None and conn.is_connected():
    #         conn.close()
    #         print('Connection closed | answer added! ')


    """ Connect to Postgresql database """
    db_config = read_db_config()
    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        sql = "INSERT INTO daily_answers (chat_id, date_answer, time_answer, username, first_name, last_name, location, event, event_description) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        cursor.execute(sql, (str(chat_id), str(date_answer), str(time_answer), str(username), str(first_name), str(last_name), str(location), str(event), str(event_description)))

        conn.commit()
    except psycopg2.OperationalError as error:
        print("Answer not added: ", str(chat_id), " ", str(date_answer), " ", str(time_answer), " ", str(username), " ", str(first_name), " ", str(last_name), " ", str(location), " ", str(event), " ", str(event_description))
    finally:
        if conn is not None:
            conn.close()
            print('Connection closed | answer added! ')


    reply_to_user = "🔸" + str(date.today()) + " " + str(call.message.text) + " : *" + str(call.data) + "*  "
    bot.send_message(call.message.chat.id, reply_to_user,
                     parse_mode = telegram.ParseMode.MARKDOWN)



@bot.message_handler(commands=["newsletter"])
def answer(message):
    if (message.from_user.id == 809104544):
        newsletter = '✌️ *Привіт*'

        """ Connect to MySQL database """
        db_config = read_db_config()
        conn = None
        try:
            conn = MySQLConnection(**db_config)
            cursor=conn.cursor()
            sql = "SELECT DISTINCT REPLACE(`chat_id`, '.0', '')  FROM `daily_answers`;"
            cursor.execute(sql)

            myresult = cursor.fetchall()

            i = 0
            for x in myresult:
                chat_id = str(x).split("('")[1].split(".0")[0]

                i = i + 1
                try:
                    if ( chat_id != "-1001746503273" and chat_id != "nan',)"):
                        bot.send_message(chat_id,
                                         newsletter,
                                         parse_mode=telegram.ParseMode.MARKDOWN)
                        today_box.today_send(chat_id)
                        print(chat_id, "  i:", i, " | message send | done ")
                except:
                    continue

        except Error as error:
            print("Cannot sended daily message :(")
        finally:
            if conn is not None and conn.is_connected():
                conn.close()
                print('Connection closed | daily message was sended! ')



@bot.message_handler(commands=["abroad"])
def answer(message):
#    if (message.from_user.id == 809104544):
#        newsletter = '✌️ Привіт ще раз, ми побачили. що ти вибрав свою геолокаці: За кордном,  \nЧи міг би ти нам написати Країну та Місто, де ти зараз перебуваєш ? 🌐'
#        all_users = pd.read_csv('data_2.csv')

#        'event_description' == 'За кордоном'
#        'date' == '2022-04-26'
    #        'chat_id'.unique().tolist()
    if (message.from_user.id == 809104544):
        newsletter = '✌️ *Привіт*'

        """ Connect to MySQL database """
        db_config = read_db_config()
        conn = None
        try:
            conn = MySQLConnection(**db_config)
            cursor=conn.cursor()
            sql = "SELECT DISTINCT `chat_id`  FROM `daily_answers`;"
            cursor.execute(sql)

            myresult = cursor.fetchall()

            for x in myresult:
                print(x)

                chat_id = x
                try:
                    # bot.send_message(chat_id,
                    #                 newsletter,
                    #                 parse_mode=telegram.ParseMode.MARKDOWN)
                    # today_box.today_send(chat_id)
                    print(chat_id, "  i:", i, " | ABROAD QUIZ | message send | done ")
                except:
                    continue

        except Error as error:
            print("Cannot sended daily message :(")
        finally:
            if conn is not None and conn.is_connected():
                conn.close()
                print('Connection closed | Anroad Quiz was sended!  ')



# Reply to answer:
@bot.message_handler(content_types=['text'])
def read_text(message):
    #if str(message.chat.id) != str(config.get('CHAT','telegram-bot_chat')):
    #    txt = message.text
    #    print("\n", date.today(), " ", datetime.now().strftime("%H:%M:%S"))
    #    print(message.chat.username, " ", " | Answer:", txt)

    #    chat_id = message.chat.id
    #    full_name = "test"
    print(message.chat.id, " ", message.text)
    # answer = '🔻date: ' + str(date.today()) + '\n🔻time: ' + str(datetime.now().strftime("%H:%M:%S")) + \
    #              '\n🔻username: @' +  str(message.chat.username) + '\nfull name:     '
    # print(answer)
    #    bot.send_message(message.chat.id, "Дякую за твою відповідь")
    #    bot.send_message(str(config.get('CHAT','telegram-bot_chat')),
    #                     answer + "\n\nMessage:\n💭 *" + txt + "*",
    #                     parse_mode=telegram.ParseMode.MARKDOWN)
    chat_id = message.chat.id
    date_answer = date.today()
    time_answer = datetime.now().strftime("%H:%M:%S")

    event_description = message.text
    event = str(today_box.saved_question)

    # username = message.chat.username
    # first_name = message.chat.first_name
    # last_name = message.chat.last_name
    # location = "None"



    # """ Connect to MySQL database """
    # db_config = read_db_config()
    # conn = None
    # try:
    #     conn = MySQLConnection(**db_config)
    #     cursor = conn.cursor()
    #
    #     sql = "INSERT INTO `daily_answers` (`chat_id`, `date_answer`, `time_answer`, `event`, `event_description`) VALUES (%s, %s, %s, %s, %s)"
    #     cursor.execute(sql, (
    #     str(chat_id), str(date_answer), str(time_answer),
    #     str(event), str(event_description)))
    #
    #     conn.commit()
    # except Error as error:
    #     print("Answer not added: ", str(chat_id), " ", str(date_answer), " ", str(time_answer), " ", str(event), " ",
    #           str(event_description))
    # finally:
    #     if conn is not None and conn.is_connected():
    #         conn.close()
    #         print('Connection closed | answer added! ')




    """ Connect to Postgresql database """
    db_config = read_db_config()
    conn = None
    try:
        conn = psycopg2.connect(**db_config)
        cursor=conn.cursor()

        sql = "INSERT INTO daily_answers (chat_id, date_answer, time_answer, event, event_description) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(sql, (
        str(chat_id), str(date_answer), str(time_answer),
        str(event), str(event_description)))

        conn.commit()
    except psycopg2.OperationalError as error:
        print("Answer not added: ", str(chat_id), " ", str(date_answer), " ", str(time_answer), " ", str(event), " ", str(event_description))
    finally:
        if conn is not None:
            conn.close()
            print('Connection closed | answer added! ')



    # reply_to_user = "🔸" + str(date.today()) + " " + str(message.text) + "*  "
    # bot.send_message(message.chat.id, str(message.text), parse_mode=telegram.ParseMode.MARKDOWN)

# run bot
bot.polling(none_stop=True, interval=0)
