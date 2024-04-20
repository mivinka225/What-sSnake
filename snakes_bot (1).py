import telebot, numpy as np, os
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image
import io


token = ''
bot = telebot.TeleBot(token)
welcome_message = '''Здравствуйте. Я могу помочь узнать вид змеи по фотографии.
На данный момент различаю 3 вида самых распространных змей России: Ужи, Гадюки, Медянки'''
wait_for_message = 'Обрабатываю'

classes = {0: 'Уж', 1: 'Гадюка', 2: 'Медянка', 3: 'Змея не найдена'}
model_path = os.path.abspath(os.getcwd()) # Путь до директиории с текущим скриптом

# Define a model
model = keras.models.load_model(model_path, custom_objects=None, compile=True, options=None)


@bot.message_handler(commands=['start'])
def welcome(message):
    bot.reply_to(message, welcome_message)


@bot.message_handler(content_types=['photo'])
def get_photo(message):
    # Get an image from chat
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    downloaded_file = bot.download_file(file_info.file_path)

    # Get a prediction from model
    answer = predict_snake(downloaded_file)

    # Send an answer to the use
    bot.reply_to(message, answer)


@bot.message_handler(func=lambda message: True)
def echo_all(message):
	bot.reply_to(message, 'Прикрепите фотографию и я попробую определить змею!')


def predict_snake(img):
    # Prepare an image
    img = Image.open(io.BytesIO(img))
    img = img.convert('RGB')
    img = img.resize((224, 224), Image.NEAREST)
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    final_image = keras.applications.densenet.preprocess_input(img_array_expanded_dims)

    # Make a prediction
    pred = np.argmax(model.predict(final_image))

    # Map number to text with shakes breed
    predicted_snake = classes[pred]
    return predicted_snake


bot.polling()
