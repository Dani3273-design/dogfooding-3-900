# -*- coding: utf-8 -*-

CITY_NAME = 'Guangzhou'

WEATHER_API_URL = f'https://wttr.in/{CITY_NAME}?format=j1'

INIT_HISTORY_DAYS = 30

SEQUENCE_DAYS = 7

TRAIN_EPOCHS = 50

TRAIN_BATCH_SIZE = 4

MODEL_NAME = 'weather_model.keras'

DATA_FILENAME = 'weather_data.csv'

CONFIG_FILENAME = 'config.json'
