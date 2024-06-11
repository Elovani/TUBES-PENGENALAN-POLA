import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime

# Baca dataset
data = pd.read_csv('Meta_Dataset.csv')

# Prapemrosesan Tanggal
data['Date'] = pd.to_datetime(data['Date'])
data['Days_Since_Start'] = (data['Date'] - data['Date'].min()).dt.days

# Pilih Fitur
X = data.drop(columns=['Close', 'Date'])  # Hapus kolom 'Close' dan 'Date'
y = data['Close']

# Bagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model Random Forest Regressor
model = RandomForestRegressor()

# Latih model
model.fit(X_train, y_train)

# Evaluasi model pada data uji
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Random Forest Regressor Mean Absolute Error: {mae}")
