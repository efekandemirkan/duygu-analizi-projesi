import os
import tensorflow as tf

# Temel Parametreler
IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 64

# Klasör yolları
TRAIN_DIR = os.path.join("dataset", "train")
TEST_DIR = os.path.join("dataset", "test")

def load_data():
    print("Veri seti taranıyor ve yükleniyor...")

    # Eğitim verisi
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True
    )

    # Doğrulama/Test verisi
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    class_names = train_ds.class_names

    # Normalizasyon: 0-255 piksel değerlerini 0.0-1.0 arasına ölçekler
    rescale_layer = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (rescale_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (rescale_layer(x), y))

    # Bellek optimizasyonu
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names

if __name__ == "__main__":
    train_data, val_data, classes = load_data()
    print("\n--- ÖN İŞLEME BAŞARILI ---")
    print(f"Toplam Duygu Sınıfı Sayısı: {len(classes)}")
    print(f"Sınıflar: {classes}")