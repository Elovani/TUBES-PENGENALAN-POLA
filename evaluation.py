import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime

# Baca dataset
data = pd.read_csv('Meta_Dataset.csv')

# Mengonversi kolom 'Date' menjadi format datetime
data['Date'] = pd.to_datetime(data['Date'])

# Menghitung jumlah hari sejak tanggal minimum
data['Days_Since_Start'] = (data['Date'] - data['Date'].min()).dt.days

# Pilih fitur dan label
X = data.drop(columns=['Close', 'Date'])  # Hapus kolom 'Close' dan 'Date'
y = data['Close']

# Bagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model
model = RandomForestRegressor()

# Latih model
model.fit(X_train, y_train)

# Evaluasi model dengan data uji
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (Test Set): {mae}")

# Evaluasi model dengan validasi silang (cross-validation)
cv_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=10)  # Gunakan 10 lipatan
mae_cv = -cv_scores.mean()
print(f"Mean Absolute Error (Cross-Validation): {mae_cv}")
