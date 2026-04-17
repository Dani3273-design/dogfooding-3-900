#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import csv
import datetime
import urllib.request
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    WEATHER_API_URL,
    INIT_HISTORY_DAYS,
    SEQUENCE_DAYS,
    TRAIN_EPOCHS,
    TRAIN_BATCH_SIZE,
    MODEL_NAME,
    DATA_FILENAME,
    CONFIG_FILENAME,
    CITY_NAME
)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_DIR = os.path.join(BASE_DIR, 'memory')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_FILE = os.path.join(MEMORY_DIR, DATA_FILENAME)
MODEL_FILE = os.path.join(MODEL_DIR, MODEL_NAME)
NORM_CONFIG_FILE = os.path.join(MEMORY_DIR, CONFIG_FILENAME)

WEATHER_MAP = {
    'Sunny': 0, 'Clear': 0,
    'Partly Cloudy': 1, 'Cloudy': 1, 'Overcast': 2,
    'Rain': 3, 'Light Rain': 3, 'Moderate Rain': 3, 'Heavy Rain': 3,
    'Snow': 4,
    'Fog': 5, 'Mist': 5,
    'Thunderstorm': 6
}

WEATHER_TEXT_MAP = {
    0: 'Sunny',
    1: 'Partly Cloudy',
    2: 'Overcast',
    3: 'Rain',
    4: 'Snow',
    5: 'Fog/Mist',
    6: 'Thunderstorm'
}

def init_dirs():
    os.makedirs(MEMORY_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['fetch_time', 'date', 'temp_max', 'temp_min', 'weather', 'humidity', 'windspeed', 'wind_degree', 'pressure', 'visibility'])

def get_wttr_weather():
    try:
        print(f"Fetching weather data for {CITY_NAME} from wttr.in...")
        with urllib.request.urlopen(WEATHER_API_URL, timeout=15) as response:
            data = json.loads(response.read().decode())
        return data
    except Exception as e:
        print(f"API request failed: {e}")
        return None

def parse_weather_data(data):
    if not data or 'weather' not in data:
        return []
    
    weather_list = []
    
    for day_data in data['weather']:
        date_str = day_data['date']
        hourly = day_data.get('hourly', [{}])
        
        weather_desc = hourly[0].get('weatherDesc', [{}])[0].get('value', 'Unknown')
        
        weather_code = 1
        for key, val in WEATHER_MAP.items():
            if key.lower() in weather_desc.lower():
                weather_code = val
                break
        
        weather_list.append({
            'date': date_str,
            'temp_max': float(day_data.get('maxtempC', 25)),
            'temp_min': float(day_data.get('mintempC', 18)),
            'weather': weather_code,
            'humidity': float(hourly[0].get('humidity', 65)),
            'windspeed': float(hourly[0].get('windspeedKmph', 10)),
            'wind_degree': float(hourly[0].get('winddirDegree', 0)),
            'pressure': float(hourly[0].get('pressure', 1013)),
            'visibility': float(hourly[0].get('visibility', 10))
        })
    
    return weather_list

def generate_simulate_data(days=30):
    print(f"Generating {days} days of simulated weather data...")
    simulate_data = []
    base_date = datetime.date.today() - datetime.timedelta(days=days)
    
    np.random.seed(42)
    
    for i in range(days):
        current_date = base_date + datetime.timedelta(days=i)
        
        month = current_date.month
        if 5 <= month <= 10:
            base_temp_max = 28 + np.random.randn() * 4
            base_temp_min = 22 + np.random.randn() * 3
        else:
            base_temp_max = 20 + np.random.randn() * 5
            base_temp_min = 12 + np.random.randn() * 4
        
        simulate_data.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'temp_max': np.clip(base_temp_max, 5, 40),
            'temp_min': np.clip(base_temp_min, 0, 30),
            'weather': np.random.choice([0, 1, 2, 3], p=[0.4, 0.35, 0.15, 0.1]),
            'humidity': 65 + np.random.randn() * 15,
            'windspeed': 10 + np.random.randn() * 5,
            'wind_degree': np.random.uniform(0, 360),
            'pressure': 1013 + np.random.randn() * 5,
            'visibility': 10 + np.random.randn() * 3
        })
    
    return simulate_data

def save_weather_data(weather_list):
    if not weather_list:
        return 0
    
    existing_dates = set()
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_dates.add(row['date'])
    
    new_count = 0
    fetch_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(DATA_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for w in weather_list:
            if w['date'] not in existing_dates:
                writer.writerow([
                    fetch_time,
                    w['date'],
                    w['temp_max'],
                    w['temp_min'],
                    w['weather'],
                    w['humidity'],
                    w['windspeed'],
                    w['wind_degree'],
                    w['pressure'],
                    w['visibility']
                ])
                new_count += 1
                existing_dates.add(w['date'])
    
    return new_count

def update_weather_data():
    print("=" * 50)
    print("Updating weather data")
    print("=" * 50)
    
    data_count = get_data_count()
    need_init = data_count < INIT_HISTORY_DAYS
    
    if need_init:
        print(f"Insufficient data ({data_count} < {INIT_HISTORY_DAYS}), initializing historical data...")
        simulate_data = generate_simulate_data(INIT_HISTORY_DAYS)
        count = save_weather_data(simulate_data)
        print(f"Initialization complete. Added {count} historical records.")
    
    wttr_data = get_wttr_weather()
    if wttr_data:
        weather_list = parse_weather_data(wttr_data)
        count = save_weather_data(weather_list)
        print(f"Fetched from wttr.in. Added {count} new weather records.")
    else:
        print("API fetch failed. Continuing with existing data.")
    
    final_count = get_data_count()
    print(f"\nTotal weather records: {final_count}")
    return final_count

def get_data_count():
    if not os.path.exists(DATA_FILE):
        return 0
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return sum(1 for line in f) - 1

def load_all_data():
    data = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    data.sort(key=lambda x: x['date'])
    
    dates = []
    features = []
    for row in data:
        dates.append(row['date'])
        feat = [
            float(row['temp_max']),
            float(row['temp_min']),
            float(row['weather']),
            float(row['humidity']),
            float(row['windspeed']),
            float(row['wind_degree']),
            float(row['pressure']),
            float(row['visibility'])
        ]
        features.append(feat)
    
    return dates, np.array(features, dtype=np.float32)

def create_model(input_shape):
    model = keras.Sequential([
        layers.LSTM(64, input_shape=input_shape, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(8)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_sequences(data, seq_length=SEQUENCE_DAYS):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_model():
    print("\n" + "=" * 50)
    print("Training weather prediction model")
    print("=" * 50)
    
    dates, features = load_all_data()
    
    if len(features) < SEQUENCE_DAYS * 2:
        print(f"Insufficient data. Required: {SEQUENCE_DAYS * 2}, Current: {len(features)}")
        return False
    
    print(f"Training data: {len(features)} days, Features: {features.shape[1]}")
    
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1
    features_norm = (features - mean) / std
    
    X, y = create_sequences(features_norm, seq_length=SEQUENCE_DAYS)
    
    if len(X) < 5:
        print(f"Insufficient sequence data: {len(X)} records")
        return False
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Train set: {len(X_train)} records, Test set: {len(X_test)} records")
    print(f"Sequence length: {SEQUENCE_DAYS} days")
    
    model = create_model((SEQUENCE_DAYS, 8))
    print("\nStarting training...")
    
    history = model.fit(
        X_train, y_train,
        epochs=TRAIN_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        validation_split=0.2,
        verbose=1
    )
    
    model.save(MODEL_FILE)
    
    config = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'sequence_days': SEQUENCE_DAYS,
        'feature_count': features.shape[1],
        'last_trained': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(NORM_CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTraining complete!")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Model saved to: {MODEL_FILE}")
    return True

def predict_weather(target_date_str):
    print("\n" + "=" * 50)
    print("Weather Prediction")
    print("=" * 50)
    
    if not os.path.exists(MODEL_FILE) or not os.path.exists(NORM_CONFIG_FILE):
        print("ERROR: Model not found. Please run: python main.py train first")
        return False
    
    with open(NORM_CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    mean = np.array(config['mean'])
    std = np.array(config['std'])
    seq_days = config.get('sequence_days', SEQUENCE_DAYS)
    
    model = keras.models.load_model(MODEL_FILE)
    
    dates, features = load_all_data()
    
    if len(features) < seq_days:
        print(f"ERROR: Insufficient data. Need {seq_days} days minimum.")
        return False
    
    today = datetime.date.today()
    if target_date_str.lower() == 'tomorrow':
        target_date = today + datetime.timedelta(days=1)
    elif target_date_str.lower() == 'dayafter':
        target_date = today + datetime.timedelta(days=2)
    else:
        try:
            target_date = datetime.datetime.strptime(target_date_str, '%Y-%m-%d').date()
        except:
            print("ERROR: Invalid date format. Use 'tomorrow', 'dayafter', or 'YYYY-MM-DD'")
            return False
    
    days_diff = (target_date - today).days
    if days_diff < 1 or days_diff > 14:
        print("ERROR: Can only predict 1-14 days into the future")
        return False
    
    print(f"Predicting based on past {seq_days} days of data...")
    
    last_days = features[-seq_days:]
    current_seq = (last_days - mean) / std
    result_features = last_days.copy()
    
    for i in range(days_diff):
        input_seq = current_seq[-seq_days:].reshape(1, seq_days, 8)
        pred_norm = model.predict(input_seq, verbose=0)[0]
        pred = pred_norm * std + mean
        result_features = np.vstack([result_features, pred])
        current_seq = np.vstack([current_seq, pred_norm])
    
    pred = result_features[-1]
    weather_code = int(np.clip(round(pred[2]), 0, 6))
    weather_text = WEATHER_TEXT_MAP.get(weather_code, 'Unknown')
    
    print("")
    print(f"Prediction Date: {target_date.strftime('%Y-%m-%d')}")
    print(f"City: {CITY_NAME}")
    print("-" * 50)
    print(f"Max Temperature: {pred[0]:.1f} C")
    print(f"Min Temperature: {pred[1]:.1f} C")
    print(f"Weather: {weather_text}")
    print(f"Humidity: {pred[3]:.0f} %")
    print(f"Wind Speed: {pred[4]:.1f} km/h")
    print(f"Wind Direction: {pred[5]:.0f} deg")
    print(f"Pressure: {pred[6]:.0f} hPa")
    print(f"Visibility: {pred[7]:.1f} km")
    print("=" * 50)
    
    return True

def show_data_info():
    print("\n" + "=" * 50)
    print("Data Information")
    print("=" * 50)
    
    dates, features = load_all_data()
    print(f"Total weather records: {len(dates)} days")
    
    if len(dates) > 0:
        print(f"Date range: {dates[0]} to {dates[-1]}")
        print(f"\nData source: wttr.in ({CITY_NAME})")
        print(f"\nLast 7 days:")
        for d, f in list(zip(dates, features))[-7:]:
            weather_text = WEATHER_TEXT_MAP.get(int(round(f[2])), 'Unknown')
            print(f"  {d}: Max {f[0]:.0f}C / Min {f[1]:.0f}C / {weather_text}")
    else:
        print("No data available. Run: python main.py update")
    
    print("=" * 50)

def print_help():
    print("Weather Prediction ML System")
    print("=" * 50)
    print("Usage:")
    print("  python main.py update          # Update weather data only")
    print("  python main.py train           # Train model (auto-update data first)")
    print("  python main.py tomorrow        # Predict tomorrow's weather")
    print("  python main.py dayafter        # Predict day after tomorrow")
    print("  python main.py 2026-04-20      # Predict specific date")
    print("  python main.py info            # Show data information")
    print("  python main.py auto            # Update data AND train model")
    print("=" * 50)

def main():
    init_dirs()
    
    if len(sys.argv) < 2:
        print_help()
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == 'update':
        update_weather_data()
    elif cmd == 'train':
        update_weather_data()
        train_model()
    elif cmd == 'info':
        show_data_info()
    elif cmd == 'auto':
        update_weather_data()
        train_model()
    elif cmd in ['tomorrow', 'dayafter'] or (len(cmd) == 10 and cmd.count('-') == 2):
        predict_weather(sys.argv[1])
    else:
        print(f"Unknown command: {cmd}")
        print("Run python main.py for help")

if __name__ == '__main__':
    main()
