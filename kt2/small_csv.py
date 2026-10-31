import pandas as pd

# Названия файлов
input_file = r'C:\Users\musai\.cache\kagglehub\datasets\sobhanmoosavi\us-accidents\versions\13\US_Accidents_March23.csv'
output_file = 'not_so_short_file.csv'

# Читаем только первые 50000 строк (не считая заголовок)
df = pd.read_csv(input_file, nrows=2000000)

# Сохраняем в новый файл
df.to_csv(output_file, index=False)

print(f"Файл успешно сокращен до {len(df)} строк и сохранен как {output_file}")