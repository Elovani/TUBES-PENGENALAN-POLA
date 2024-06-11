import pandas as pd

# Baca dataset
data = pd.read_csv('Meta_Dataset.csv')  # Ganti 'saham_data.csv' dengan nama file Anda

# Mengekstraksi ciri-ciri
def extract_features(data):
    features = []
    # Ciri-ciri statistik sederhana
    features.append(data['Close'].mean())  # Rata-rata harga penutupan
    features.append(data['Close'].std())   # Standar deviasi harga penutupan
    features.append(data['Volume'].mean()) # Rata-rata volume perdagangan
    features.append(data['Volume'].std())  # Standar deviasi volume perdagangan
    # Tambahan ciri-ciri lainnya sesuai kebutuhan Anda
    return features

# Ekstraksi ciri untuk dataset secara keseluruhan
all_features = extract_features(data)

# Tampilkan hasil ekstraksi ciri
print("Ciri yang diekstraksi:")
print(all_features)
