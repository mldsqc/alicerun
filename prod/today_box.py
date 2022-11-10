from random import choice
import telebot as telebot
import telegram
import os.path
import configparser

from telebot import types

config = configparser.ConfigParser()
config.read('/src/Telegram_bot/config.ini')

bot = telebot.TeleBot(config.get('telegram-bot', 'TOKEN'))


quest = ['34351', '44531', '11111', '44444', '16', '3246', '4652']
saved_question = 'АНУКА ВОПРОСИК' + str(choice(quest))

def today_send(chat_id):
    # bot.delete_message(m.chat.id, m.message_id)
    markup1 = types.InlineKeyboardMarkup()
    buttonA1 = types.InlineKeyboardButton('Full-time', callback_data='Full-time')
    buttonB1 = types.InlineKeyboardButton('Part-time', callback_data='Part-time')
    buttonC1 = types.InlineKeyboardButton("Не можу працювати", callback_data="Не можу працювати")

    markup2 = types.InlineKeyboardMarkup()
    buttonA2 = types.InlineKeyboardButton("0-2", callback_data="0-2 😟")
    buttonB2 = types.InlineKeyboardButton("2-5", callback_data="2-5 😐")
    buttonC2 = types.InlineKeyboardButton("5-7", callback_data="5-7 🙂")
    buttonD2 = types.InlineKeyboardButton("8-10", callback_data="8-10 🤗")

    markup3 = types.InlineKeyboardMarkup()
    buttonA3 = types.InlineKeyboardButton('Так, напишіть мені', callback_data='Так, напишіть мені')
    buttonB3 = types.InlineKeyboardButton('Ні, все ок', callback_data='Ні, все ок')

    markup4 = types.InlineKeyboardMarkup()
    buttonA4 = types.InlineKeyboardButton('admiration', callback_data='admiration')
    buttonB4 = types.InlineKeyboardButton("adoration", callback_data="adoration")
    buttonC4 = types.InlineKeyboardButton('aesthetic', callback_data='aesthetic')
    buttonD4 = types.InlineKeyboardButton("appreciation", callback_data="appreciation")
    buttonE4 = types.InlineKeyboardButton("amusement", callback_data="amusement")
    buttonF4 = types.InlineKeyboardButton("anger", callback_data="anger")
    buttonG4 = types.InlineKeyboardButton("anxiety", callback_data="anxiety")
    buttonH4 = types.InlineKeyboardButton("awe", callback_data="awe")
    buttonI4 = types.InlineKeyboardButton("awkwardness", callback_data="awkwardness")
    buttonJ4 = types.InlineKeyboardButton("boredom", callback_data="boredom")
    buttonK4 = types.InlineKeyboardButton("calmness", callback_data="calmness")
    buttonL4 = types.InlineKeyboardButton("confusion", callback_data="confusion")
    buttonM4 = types.InlineKeyboardButton("craving", callback_data="craving")
    buttonN4 = types.InlineKeyboardButton("disgust", callback_data="disgust")
    buttonO4 = types.InlineKeyboardButton("empathic", callback_data="empathic")
    buttonP4 = types.InlineKeyboardButton("pain", callback_data="pain")
    buttonQ4 = types.InlineKeyboardButton("entrancement", callback_data="entrancement")
    buttonR4 = types.InlineKeyboardButton("excitement", callback_data="excitement")
    buttonS4 = types.InlineKeyboardButton("+fear", callback_data="+fear")
    buttonS41 = types.InlineKeyboardButton("-fear", callback_data="-fear")
    buttonT4 = types.InlineKeyboardButton("horror", callback_data="horror")
    buttonU4 = types.InlineKeyboardButton("interest", callback_data="interest")
    buttonV4 = types.InlineKeyboardButton("joy", callback_data="joy")
    buttonW4 = types.InlineKeyboardButton("nostalgia", callback_data="nostalgia")
    buttonX4 = types.InlineKeyboardButton("relief", callback_data="relief")
    buttonY4 = types.InlineKeyboardButton("romance", callback_data="romance")
    buttonZ4 = types.InlineKeyboardButton("+sadness", callback_data="+sadness")
    buttonZ41 = types.InlineKeyboardButton("satisfaction", callback_data="satisfaction")
    buttonZ42 = types.InlineKeyboardButton("-sadness", callback_data="-sadness")
    buttonZ43 = types.InlineKeyboardButton("sexual desire", callback_data="sexual desire")
    buttonZ44 = types.InlineKeyboardButton("surprise", callback_data="surprise")
    buttonY41 = types.InlineKeyboardButton("desire", callback_data="desire")
    buttonY42 = types.InlineKeyboardButton("happiness", callback_data="happiness")
    buttonY43 = types.InlineKeyboardButton("pride", callback_data="pride")
    buttonY44 = types.InlineKeyboardButton("inspiration", callback_data="inspiration")
    buttonY45 = types.InlineKeyboardButton("fascination", callback_data="fascination")
    buttonY46 = types.InlineKeyboardButton("anger", callback_data="anger")
    buttonY47 = types.InlineKeyboardButton("disgust", callback_data="disgust")
    buttonY48 = types.InlineKeyboardButton("-surprise", callback_data="-surprise")
    buttonY49 = types.InlineKeyboardButton("+surprise", callback_data="+surprise")
    buttonY411 = types.InlineKeyboardButton("dissatisfaction", callback_data="dissatisfaction")
    buttonY412 = types.InlineKeyboardButton("embarrassed", callback_data="embarrassed")
    buttonY413 = types.InlineKeyboardButton("indignation", callback_data="indignation")
    buttonY414 = types.InlineKeyboardButton("burnouted", callback_data="burnouted")
    buttonY415 = types.InlineKeyboardButton("emotionally balanced", callback_data="emotionally balanced")
    buttonY416 = types.InlineKeyboardButton("emotionally UNbalanced", callback_data="emotionally UNbalanced")
    buttonY417 = types.InlineKeyboardButton("pleasure from done tasks", callback_data="pleasure from done tasks")
    buttonY418 = types.InlineKeyboardButton("UNPREDICTABLE EMOTIONAL WOWs", callback_data="UNPREDICTABLE EMOTIONAL WOWs")
    buttonY419 = types.InlineKeyboardButton("motivated", callback_data="motivated")
    buttonY420 = types.InlineKeyboardButton("focused", callback_data="focused")
    buttonY421 = types.InlineKeyboardButton("procrastinated", callback_data="procrastinated")
    buttonY422 = types.InlineKeyboardButton("productive", callback_data="productive")

    # admiration, adoration, aesthetic
    # appreciation, amusement, anger, anxiety, awe, awkwardness, boredom, calmness,
    # confusion, craving, disgust, empathic
    # pain, entrancement, excitement, +fear, -fear,  horror, interest, joy, nostalgia, relief,
    # romance, sadness, satisfaction, sexual
    # desire, surprise

    # восхищение, обожание, эстетика
    # признательность, веселье, гнев, тревога, трепет, неловкость, скука, спокойствие,
    # замешательство, страстное желание, отвращение, эмпатия
    # боль, восхищение, волнение, страх, ужас, интерес, радость, ностальгия, облегчение,
    # романтика, печаль, удовлетворение, сексуальность
    # желание, удивление

    # desire  pride happiness inspiration fascination anger disgust -surprise +surprise dissatisfaction
    # embarrassed indignation
    # желание гордость счастье вдохновение очарование гнев отвращение -удивление +удивление
    # неудовлетворенность
    # смущенное возмущение

    markup4.row(buttonA4, buttonB4, buttonC4)
    markup4.row(buttonD4, buttonE4, buttonF4)
    markup4.row(buttonG4, buttonH4, buttonI4)
    markup4.row(buttonJ4, buttonK4, buttonL4)
    markup4.row(buttonP4, buttonQ4, buttonR4)
    markup4.row(buttonM4, buttonN4, buttonO4)
    markup4.row(buttonS4, buttonS41, buttonU4)
    markup4.row(buttonV4, buttonW4, buttonX4)
    markup4.row(buttonY4, buttonZ4, buttonZ41)
    markup4.row(buttonZ42, buttonZ43, buttonZ44)
    markup4.row(buttonT4, buttonY41, buttonY42)
    markup4.row(buttonY43, buttonY44, buttonY45)
    markup4.row(buttonY46, buttonY47, buttonY48)
    markup4.row(buttonY49, buttonY411, buttonY412, buttonY413)
    markup4.row(buttonY414, buttonY415, buttonY416, buttonY417)
    markup4.row(buttonY418, buttonY419, buttonY420)
    markup4.row(buttonY421, buttonY422)






    #habbits input
    markup5 = types.InlineKeyboardMarkup()
    buttonA5 = types.InlineKeyboardButton('made smth for selfefficiency', callback_data='made smth for selfefficiency')
    buttonB5 = types.InlineKeyboardButton("consumed vs generated out in the world", callback_data="consumed vs generated out in the world")
    buttonC5 = types.InlineKeyboardButton('running on plans feeling', callback_data='running on plans feeling')
    buttonD5 = types.InlineKeyboardButton("psycho practices", callback_data="psycho practices")
    buttonE5 = types.InlineKeyboardButton("5 minute journal", callback_data="5 minute journal")
    buttonF5 = types.InlineKeyboardButton("meditation", callback_data="meditation")
    buttonG5 = types.InlineKeyboardButton("look inside 4 feelings on whole life", callback_data="look inside 4 feelings on whole life")
    buttonH5 = types.InlineKeyboardButton("bucket list", callback_data="bucket list")
    buttonI5 = types.InlineKeyboardButton("wish list", callback_data="wish list")
    buttonJ5 = types.InlineKeyboardButton("financial reduce costs", callback_data="financial reduce costs")
    buttonK5 = types.InlineKeyboardButton("encresed income", callback_data="encresed income")
    buttonL5 = types.InlineKeyboardButton("investing", callback_data="investing")

    buttonM5 = types.InlineKeyboardButton("new opensource", callback_data="new opensource")
    buttonN5 = types.InlineKeyboardButton("opensourced work", callback_data="opensourced work")
    buttonO5 = types.InlineKeyboardButton("opensourced questions answered", callback_data="opensourced questions answered")

    buttonP5 = types.InlineKeyboardButton("slept_GOOD", callback_data="slept_GOOD")
    buttonQ5 = types.InlineKeyboardButton("cold shower", callback_data="cold shower")
    buttonR5 = types.InlineKeyboardButton("10 pushups everyday", callback_data="10 pushups everyday")
    buttonS5 = types.InlineKeyboardButton("big physical activity", callback_data="big physical activity")

    buttonS51 = types.InlineKeyboardButton("traveled", callback_data="traveled")
    buttonT5 = types.InlineKeyboardButton("drugs", callback_data="drugs")
    buttonU5 = types.InlineKeyboardButton("jrk", callback_data="jrk")
    buttonV5 = types.InlineKeyboardButton("porn", callback_data="porn")
    buttonW5 = types.InlineKeyboardButton("too much movies", callback_data="too much movies")
    buttonX5 = types.InlineKeyboardButton("too much youtube", callback_data="too much youtube")
    buttonY5 = types.InlineKeyboardButton("too much social media", callback_data="too much social media")
    buttonZ5 = types.InlineKeyboardButton("too much news", callback_data="too much news")

    buttonZ51 = types.InlineKeyboardButton("any pain", callback_data="any pain")
    buttonZ52 = types.InlineKeyboardButton("reading", callback_data="reading")
    buttonZ53 = types.InlineKeyboardButton("studing", callback_data="studing")
    buttonZ54 = types.InlineKeyboardButton("chess", callback_data="chess")
    buttonY51 = types.InlineKeyboardButton("ankicards", callback_data="ankicards")
    buttonY52 = types.InlineKeyboardButton("languages", callback_data="languages")

    buttonY53 = types.InlineKeyboardButton("+youtubed", callback_data="+youtubed")
    buttonY54 = types.InlineKeyboardButton("film", callback_data="film")
    buttonY55 = types.InlineKeyboardButton("tvshow", callback_data="tvshow")
    buttonY56 = types.InlineKeyboardButton("cinema", callback_data="cinema")
    buttonY57 = types.InlineKeyboardButton("gaming", callback_data="gaming")

    buttonY58 = types.InlineKeyboardButton("over_eated", callback_data="over_eated")
    buttonY59 = types.InlineKeyboardButton("fastfood", callback_data="fastfood")
    buttonY511 = types.InlineKeyboardButton("cafe", callback_data="cafe")

    buttonY512 = types.InlineKeyboardButton("social offline", callback_data="social offline")
    buttonY513 = types.InlineKeyboardButton("new people", callback_data="new people")
    buttonY514 = types.InlineKeyboardButton("old_friends", callback_data="old_friends")
    buttonY515 = types.InlineKeyboardButton("sex", callback_data="sex")
    buttonY516 = types.InlineKeyboardButton("new sex partner", callback_data="new sex partner")
    buttonY517 = types.InlineKeyboardButton("new sex practices", callback_data="new sex practices")
    buttonY518 = types.InlineKeyboardButton("harmony_pleasurefull", callback_data="critiqued_by_HER")
    buttonY519 = types.InlineKeyboardButton("critiqued_by_ME", callback_data="critiqued_by_ME")
    buttonY520 = types.InlineKeyboardButton("arguing", callback_data="arguing")
    buttonY521 = types.InlineKeyboardButton("work thru complicated situations", callback_data="work thru complicated "
                                                                                             "situations")
    buttonY522 = types.InlineKeyboardButton("common goals completion", callback_data="common goals completion")


    markup5.row(buttonA5, buttonB5, buttonC5)
    markup5.row(buttonD5, buttonE5, buttonF5)
    markup5.row(buttonG5, buttonH5, buttonI5)
    markup5.row(buttonJ5, buttonK5, buttonL5)
    markup5.row(buttonP5, buttonQ5, buttonR5)
    markup5.row(buttonM5, buttonN5, buttonO5)
    markup5.row(buttonS5, buttonS51, buttonU5)
    markup5.row(buttonV5, buttonW5, buttonX5)
    markup5.row(buttonY5, buttonZ5, buttonZ51)
    markup5.row(buttonZ52, buttonZ53, buttonZ54)
    markup5.row(buttonT5, buttonY51, buttonY52)
    markup5.row(buttonY53, buttonY54, buttonY55)
    markup5.row(buttonY56, buttonY57, buttonY58)
    markup5.row(buttonY59, buttonY511, buttonY512, buttonY513)
    markup5.row(buttonY514, buttonY515, buttonY516)
    markup5.row(buttonY517, buttonY518, buttonY519)
    markup5.row(buttonY520, buttonY521, buttonY522)






    bot.send_message(chat_id,
                     "✌ ️ *ЧОКАК ГОПНИЧЕГГ*? \n\n  ДАВАЙ ПРОЧЕКАЕМСЯ",
                     parse_mode=telegram.ParseMode.MARKDOWN)

    bot.send_message(chat_id,
                     "🔹 1. ЧОКАК ЧЕКАЕМ ЭМОЙШЕНЗ:",
                     parse_mode=telegram.ParseMode.MARKDOWN,
                     reply_markup=markup4)

    bot.send_message(chat_id,
                     "🔹 1. ЧОКАК ЧЕКАЕМ ХЭБИТЗ:",
                     parse_mode=telegram.ParseMode.MARKDOWN,
                     reply_markup=markup5)



    bot.send_message(chat_id,
                     "🔹 2. ВЗЪЕБЕМ ВСЕХ *НА РАБОТЕ*?",
                     parse_mode=telegram.ParseMode.MARKDOWN,
                     reply_markup=markup1.row(buttonA1, buttonB1, buttonC1))

    bot.send_message(chat_id,
                     "🔹 3. Мій *стан* від 0 до 10 сьогодні: ",
                     parse_mode=telegram.ParseMode.MARKDOWN,
                     reply_markup=markup2.row(buttonA2, buttonB2, buttonC2, buttonD2))





    #daily questionaring
    bot.send_message(chat_id,
                     saved_question,
                     parse_mode=telegram.ParseMode.MARKDOWN)
