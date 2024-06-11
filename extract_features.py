import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Baca dataset
data = pd.read_csv('Meta_Dataset.csv')  # Ganti 'nama_file.csv' dengan nama file Anda

# Pemrosesan data
data['Date'] = pd.to_datetime(data['Date'])  # Konversi kolom tanggal ke tipe data datetime
data.set_index('Date', inplace=True)  # Set kolom tanggal sebagai indeks

# Normalisasi data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Ekstraksi ciri
# Misalnya, kita akan menggunakan harga penutupan (Close) sebagai fitur
features = scaled_data[:, 3]  # Index 3 adalah kolom harga penutupan

# Contoh ekstraksi ciri lainnya:
# Jika Anda ingin menggunakan beberapa fitur, Anda dapat menentukan mereka di sini
# features = scaled_data[:, [3, 4, 5]]  # Menggunakan Close, Adj Close, Volume sebagai fitur

# Tampilkan hasil ekstraksi ciri
print("Ciri yang diekstraksi:")
print(features)
