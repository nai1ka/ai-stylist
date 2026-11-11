# Примечания к моделям

## Платформа

- Машина разработки: macOS 15.3.1, Apple Silicon arm64.
- Системный Python: 3.14.10; также найден Python 3.11.9, выбран для проекта.
- MediaPipe официально указывает macOS arm64 и Python 3.9–3.14: https://developers.google.com/edge/mediapipe/solutions/setup_python (проверено 2026-11-10).

## Face Landmarker

- Кандидат: официальный MediaPipe Face Landmarker, Tasks API, режим IMAGE, `num_faces=2`, CPU.
- URL загрузчика: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
- Скачивание и импорт пока не подтверждены в окружении.

## Сегментация кожи, волос и глаз

- Модель: `jonathandinu/face-parsing`, SegFormer fine-tuned для face parsing на CelebAMask-HQ. Карточка: https://huggingface.co/jonathandinu/face-parsing
- Загрузчик фиксирует ревизию `8ee285cb01328e6b6d0d33f2cdbc6d8505ac069d`; config этой ревизии содержит `skin.png`, `hair.png`, `l_eye.png`, `r_eye.png` в `id2label`.
- Классы глаз — области глаз, не отдельные радужки. Приложение явно показывает это ограничение.
- Инференс запускается на CPU; processor использует родную предобработку модели, а logits интерполируются к размеру портрета.
- Лицензия и запуск на фото пользователя требуют отдельной фактической проверки.

## Классификатор формы лица

- Кандидат: `metadome/face_shape_classification` — ViT, карточка перечисляет Heart, Oblong, Oval, Round, Square: https://huggingface.co/metadome/face_shape_classification
- Ревизия, лицензия, processor и фактические метки config не зафиксированы: сеть в среде разработки недоступна.
- Предсказания и реальный инференс не проверены. До проверки нельзя считать этап моделей завершённым.
- Crop строится по min/max landmarks с расширением 20% на изображении, подготовленном `images.load_image`; это эвристика приложения.
