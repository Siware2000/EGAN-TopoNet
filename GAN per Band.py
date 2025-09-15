import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, Conv2DTranspose, Conv2D, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === SETTINGS ===
root_dir = "C:/Users/Siwar/OneDrive/Desktop/My research paper topic/EGAN-TopoNet/Topomap GAN Split/train"
output_dir = "C:/Users/Siwar/OneDrive/Desktop/My research paper topic/EGAN-TopoNet/Synthetic Topomaps"
bands = ["theta", "alpha", "beta"]
image_size = (64, 64)
latent_dim = 100
batch_size = 32
epochs = 3000
sample_interval = 500

# === Build Generator ===
def build_generator():
    model = Sequential()
    model.add(Dense(128 * 16 * 16, activation="relu", input_dim=latent_dim))
    model.add(Reshape((16, 16, 128)))
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation='tanh'))
    return model

# === Build Discriminator ===
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=4, strides=2, padding="same", input_shape=(64, 64, 1)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# === Training Loop ===
def train_gan_for_band(band):
    print(f"🎯 Training GAN for {band} band")
    band_dir = os.path.join(root_dir, band)
    output_band_dir = os.path.join(output_dir, band)
    os.makedirs(output_band_dir, exist_ok=True)

    datagen = ImageDataGenerator(rescale=1. / 255)
    # Check if the band folder has images
    if not os.path.exists(band_dir) or len(os.listdir(band_dir)) == 0:
        print(f"⚠️ Skipping {band}: No images found in {band_dir}")
        return

    # Load data
    data_flow = datagen.flow_from_directory(
        root_dir,
        target_size=image_size,
        color_mode='grayscale',
        batch_size=batch_size,
        classes=[band],
        class_mode=None,
        shuffle=True
    )



    # Build and compile models
    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

    discriminator.trainable = False
    gan = Sequential([generator, discriminator])
    gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # === Training Loop ===
    for epoch in range(epochs + 1):
        real_imgs = next(data_flow)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        g_loss = gan.train_on_batch(noise, valid)

        if epoch % sample_interval == 0:
            print(f"[{band}] Epoch {epoch} | D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}% | G loss: {g_loss:.4f}")
            sample_img = generator.predict(np.random.normal(0, 1, (1, latent_dim)))[0]
            sample_img = 0.5 * sample_img + 0.5  # scale [-1, 1] to [0, 1]
            plt.imshow(sample_img[:, :, 0], cmap='gray')
            plt.axis('off')
            plt.title(f"{band} - Epoch {epoch}")
            plt.savefig(os.path.join(output_band_dir, f"sample_{epoch}.png"))
            plt.close()

# === RUN FOR ALL BANDS ===
for band in bands:
    train_gan_for_band(band)

print("✅ GAN training completed for all bands.")
