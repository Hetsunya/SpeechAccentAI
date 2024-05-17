import os
from PIL import Image
from fpdf import FPDF

# Путь к папке с изображениями Mel-спектрограмм
mels_folder = "mels"

# Создаем PDF документ
pdf = FPDF()

# Получаем список всех файлов в папке mels
image_files = [f for f in os.listdir(mels_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Добавляем каждую картинку на отдельную страницу PDF
for image_file in image_files:
    image_path = os.path.join(mels_folder, image_file)
    
    # Открываем изображение
    image = Image.open(image_path)

    # Добавляем страницу в PDF
    pdf.add_page()

    # Вставляем изображение на страницу
    pdf.image(image_path, x=10, y=10, w=190)  # Настраиваем размеры и позицию изображения

# Сохраняем PDF файл
pdf.output("mel_spectrograms.pdf", "F")

print("PDF файл с Mel-спектрограммами создан!")