# 天气预测机器学习模型

基于 TensorFlow/Keras LSTM 时间序列的轻量级天气预测系统，纯CPU运行，支持每日定时更新训练。

---

## 项目介绍

本项目使用LSTM神经网络进行时间序列预测，实现了一个完整的天气机器学习预测系统。

### 核心特点

- **纯CPU运行**：无需GPU，普通笔记本和服务器即可运行
- **训练快速**：CPU上50轮训练仅需1-2分钟
- **定时执行**：支持crontab每日0点自动更新数据并训练
- **简单易用**：纯命令行接口，一行命令即可查询预测
- **数据持久化**：CSV文本格式存储所有历史数据

### 预测维度

| 字段 | 说明 |
|------|------|
| 最高温度 | 当日预计最高气温(°C) |
| 最低温度 | 当日预计最低气温(°C) |
| 天气状况 | 晴/多云/阴/雨等天气类型 |
| 相对湿度 | 空气相对湿度(%) |
| 风速 | 风速(km/h) |
| 风向 | 风向角度(0-360°) |
| 气压 | 大气压强(hPa) |
| 能见度 | 能见度(km) |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 查看数据信息

```bash
python main.py info
```

### 3. 更新数据并训练模型

```bash
python main.py auto
```

### 4. 查询天气预测

```bash
# 预测明天天气
python main.py tomorrow

# 预测指定日期
python main.py 2026-04-20
```

---

## 完整命令说明

| 命令 | 说明 |
|------|------|
| `python main.py` | 显示帮助信息 |
| `python main.py update` | 仅更新最新天气数据，不训练 |
| `python main.py train` | 训练模型（会先自动更新数据） |
| `python main.py auto` | 更新数据 + 训练模型 【推荐】 |
| `python main.py info` | 查看已记录的数据统计信息 |
| `python main.py tomorrow` | 预测明天天气 |
| `python main.py dayafter` | 预测后天天气 |
| `python main.py YYYY-MM-DD` | 预测指定日期天气（支持未来1-14天） |

---

## 定时任务配置

Linux/Mac 配置 crontab 每天0点自动执行：

```bash
crontab -e
```

添加以下内容：

```bash
0 0 * * * cd /path/to/project/main && /usr/bin/python3 main.py auto >> /tmp/weather_train.log 2>&1
```

---

## 配置文件说明

所有参数集中在 `config/config.py` 中统一管理：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CITY_NAME` | `'Guangzhou'` | 预测城市名称（英文） |
| `WEATHER_API_URL` | wttr.in | 免费天气API，无需注册 |
| `INIT_HISTORY_DAYS` | `30` | 首次运行初始化多少天历史数据 |
| `SEQUENCE_DAYS` | `7` | 用过去多少天的数据预测未来 |
| `TRAIN_EPOCHS` | `50` | 模型训练轮数 |
| `TRAIN_BATCH_SIZE` | `4` | 训练批次大小 |
| `MODEL_NAME` | `weather_model.keras` | 模型保存文件名 |
| `DATA_FILENAME` | `weather_data.csv` | 历史数据文件名 |

### 配置修改示例

修改为预测北京天气：
```python
# config/config.py
CITY_NAME = 'Beijing'
```

增加训练轮数：
```python
# config/config.py
TRAIN_EPOCHS = 100
```

使用过去14天数据进行预测：
```python
# config/config.py
SEQUENCE_DAYS = 14
```

---

## 目录结构

```
main/
├── main.py              # 程序主入口
├── requirements.txt     # Python依赖
├── README.md            # 本文档
├── config/
│   ├── __init__.py      # 配置模块导出
│   └── config.py        # 全局参数配置
├── memory/
│   ├── weather_data.csv # 所有历史天气数据
│   └── config.json      # 模型归一化参数
└── model/
    └── weather_model.keras # 训练好的Keras模型
```

### 文件说明

- **config/config.py**：所有可调参数集中管理
- **weather_data.csv**：文本格式存储，可直接用Excel打开查看
- **config.json**：存储数据归一化参数，保证预测一致性
- **weather_model.keras**：Keras原生模型格式

---

## 使用示例

### 示例1：查看帮助

```bash
$ python main.py

Weather Prediction ML System
==================================================
Usage:
  python main.py update          # Update weather data only
  python main.py train           # Train model (auto-update data first)
  python main.py tomorrow        # Predict tomorrow's weather
  python main.py dayafter        # Predict day after tomorrow
  python main.py 2026-04-20      # Predict specific date
  python main.py info            # Show data information
  python main.py auto            # Update data AND train model
==================================================
```

### 示例2：训练模型

```bash
$ python main.py auto

==================================================
Updating weather data
==================================================
...
==================================================
Training weather prediction model
==================================================
Training data: 33 days, Features: 8
Train set: 20 records, Test set: 6 records
Sequence length: 7 days

Starting training...
Epoch 1/50
5/5 [==============================] - 2s 134ms/step - loss: 1.0214 - mae: 0.8623
Epoch 2/50
5/5 [==============================] - 0s 12ms/step - loss: 0.9678 - mae: 0.8375
...
Epoch 50/50
5/5 [==============================] - 0s 11ms/step - loss: 0.0187 - mae: 0.1056

Training complete!
Test Loss: 0.0872
Test MAE: 0.2145
Model saved to: model/weather_model.keras
```

### 示例3：预测明天天气

```bash
$ python main.py tomorrow

==================================================
Weather Prediction
==================================================
Predicting based on past 7 days of data...

Prediction Date: 2026-04-19
City: Guangzhou
--------------------------------------------------
Max Temperature: 27.3 C
Min Temperature: 18.7 C
Weather: Partly Cloudy
Humidity: 68 %
Wind Speed: 11.8 km/h
Wind Direction: 45 deg
Pressure: 1013 hPa
Visibility: 10.0 km
==================================================
```

### 示例4：预测指定日期

```bash
$ python main.py 2026-04-25

==================================================
Weather Prediction
==================================================
Predicting based on past 7 days of data...

Prediction Date: 2026-04-25
City: Guangzhou
--------------------------------------------------
Max Temperature: 26.1 C
Min Temperature: 19.2 C
Weather: Sunny
Humidity: 65 %
Wind Speed: 9.5 km/h
Wind Direction: 90 deg
Pressure: 1015 hPa
Visibility: 12.0 km
==================================================
```

---

## 技术栈

- **Python**: 3.9+
- **TensorFlow**: 2.15.0
- **Keras**: 内置LSTM层
- **模型结构**: 双层LSTM + MLP with Dropout
- **序列窗口**: 使用过去7天数据预测未来1天

---

## 数据源

使用 wttr.in 免费开放API获取真实天气数据：

- 无需注册
- 无需API KEY
- 国内网络可直接访问
- 自动获取未来3天预报

首次运行自动生成30天模拟历史数据，后续每日自动从API获取最新真实数据。

---

## 常见问题

**问：提示"Model not found"怎么办？**

> 请先运行 `python main.py auto` 完成一次模型训练。

**问：可以预测多少天的天气？**

> 支持预测未来1-14天的天气。

**问：数据存在哪里，可以手动编辑吗？**

> 数据存在 `memory/weather_data.csv`，是标准CSV格式，可以用任何表格软件打开编辑。

**问：训练需要多久？**

> 在现代CPU上，50轮训练大约1-2分钟即可完成。

**问：需要申请API KEY吗？**

> 完全不需要！使用 wttr.in 公开免费API，开箱即用。

**问：怎么改成预测其他城市？**

> 修改 `config/config.py` 中的 `CITY_NAME` 即可，例如 'Beijing'、'Shanghai'。
