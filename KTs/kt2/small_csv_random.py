import pandas as pd

# Названия файлов
input_file = r'C:\Users\musai\.cache\kagglehub\datasets\sobhanmoosavi\us-accidents\versions\13\US_Accidents_March23.csv'
output_file = 'random_short_file.csv'

# Загружаем весь файл
df = pd.read_csv(input_file)

# Выбираем ровно 1000 случайных строк
# Если в файле меньше 1000 строк, берем все доступные
num_rows = min(10000, len(df))
df_random = df.sample(n=num_rows, random_state=42) # random_state нужен для воспроизводимости (опционально)

# Сохраняем в новый файл
df_random.to_csv(output_file, index=False)

print(f"Выбрано случайных строк: {len(df_random)}. Файл сохранен как {output_file}")
