# SpeechAccentAI

## Описание
Этот проект направлен на разработку системы машинного обучения, способной определить акцент человека по речи с использованием MFCCs (Mel-Frequency Cepstral Coefficients) и нейронной сети.

## Датасет
Speech Accent Archive

## Технологии
- Python
- Librosa
- TensorFlow
- Scikit-learn

## Файлы

- `extract_mfcc.py`: Скрипт для предобработки аудиофайлов, извлечения MFCCs и сохранения их в формате NumPy.
- `train_model.py`: Скрипт для обучения нейронной сети на извлеченных MFCCs и сохранения обученной модели.
- `predict_accent_from_audio.py`: Скрипт для загрузки обученной модели и предсказания акцента для новых аудиофайлов.
- `extracted_mfccs.npy`: Сохраненные MFCCs из обучающего набора данных.
- `accent_labels.npy`: Сохраненные метки акцентов для обучающего набора данных.
- `label_encoder_classes.npy`: Сохраненные классы LabelEncoder для преобразования меток акцентов.
- `accent_detection_model.h5`: Обученная модель нейронной сети.

## Как использовать

### Предобработка данных
Запустите `python extract_mfcc.py`. Этот скрипт обработает аудиофайлы из датасета, извлечет MFCCs и сохранит их в файлах NumPy.

### Обучение модели
Запустите `python train_model.py`. Этот скрипт обучит нейронную сеть на извлеченных MFCCs и сохранит обученную модель в файле `accent_detection_model.h5`.

### Предсказание акцента
Запустите `python predict_accent_from_audio.py`. Этот скрипт загрузит обученную модель и позволит вам указать путь к аудиофайлу для предсказания акцента.

## Результаты
Точность модели будет зависеть от качества данных, размера датасета и выбранной архитектуры нейронной сети. В процессе экспериментов можно оптимизировать параметры модели и предобработки данных для достижения наилучших результатов.

## Дальнейшее развитие
- Использовать более сложные модели нейронных сетей, такие как LSTM или CNN, для учета временных зависимостей в данных MFCC.
- Использовать техники аугментации данных для увеличения объема обучающего набора и улучшения обобщающей способности модели.
- Интегрировать систему в веб-приложение или API для удобного использования.

## Авторы
- Hetsunya