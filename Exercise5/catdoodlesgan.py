import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

from google.colab import drive

drive.mount("/content/drive")  # Don't change this

# % ls
#
# % cd
# drive
#
# % cd
# MyDrive /
#
# % ls
#
# % pwd

cat_doodles = np.load('/content/drive/My Drive/cats.npy')

X_train = (cat_doodles.astype(np.float32) / 127.5) - 1.  # normalize to be [-1,1]
num_train = X_train.shape[0]

random_cat = X_train[np.random.randint(num_train - 1)].reshape(28, 28)
random_cat = random_cat * 127.5 + 1
plt.imshow(random_cat)  # a wild random cat appears

from tqdm import tqdm
from keras.models import Sequential, Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, Flatten
from keras.layers import Activation, Reshape
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.backend import clear_session

np.random.seed(0)

from numpy import random


class SimpleGAN():

    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.img_shape = (self.img_rows, self.img_cols, 1)
        self.latent_dim = 100

        self.optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        self.gan = self.build_GAN()

    def build_GAN(self):
        self.discriminator.trainable = False
        gan_input = Input(shape=(self.latent_dim,))
        img = self.generator(gan_input)

        gan_output = self.discriminator(img)
        gan = Model(gan_input, gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        gan.summary()

        return gan

    def build_generator(self):

        G = Sequential()

        G.add(Dense(256, input_dim=self.latent_dim, kernel_initializer=RandomNormal(stddev=0.02)))
        G.add(LeakyReLU(0.2))
        G.add(Dense(512))
        G.add(LeakyReLU(0.2))
        G.add(Dense(1024))
        G.add(LeakyReLU(0.2))
        G.add(Dense(2048))
        G.add(LeakyReLU(0.2))
        G.add(Dense(np.prod(self.img_shape), activation='tanh'))
        G.summary()

        noise = Input(shape=(self.latent_dim,))
        img = G(noise)

        return Model(noise, img)

    def build_discriminator(self):

        D = Sequential()
        D.add(Dense(1024, input_dim=np.prod(self.img_shape), kernel_initializer=RandomNormal(stddev=0.02)))
        # D.add(Dense(256, input_dim=np.prod(self.img_shape), kernel_initializer=RandomNormal(stddev=0.02)))
        D.add(LeakyReLU(0.2))
        D.add(Dropout(0.4))
        D.add(Dense(512))
        D.add(LeakyReLU(0.2))
        D.add(Dropout(0.4))
        D.add(Dense(256))
        D.add(LeakyReLU(0.2))
        D.add(Dropout(0.4))
        D.add(Dense(1, activation='sigmoid'))
        D.summary()

        img = Input(shape=(784,))
        validity = D(img)

        return Model(img, validity)

    def train(self, X_train, epochs, batch_size=128, sample_interval=50):

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        avg_losses = {'D': [], 'G': []}
        num_batches = len(X_train) // batch_size
        for epoch in range(epochs):
            d_loss_acc = 0
            g_loss_acc = 0
            for i in range(num_batches):
                # Random batch of real doodles
                imgs = X_train[np.random.randint(0, X_train.shape[0], batch_size)]

                # Generate a batch of doodles
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise)

                self.discriminator.trainable = True
                # Train the discriminator (add random smoothing to labels)
                d_loss_real = self.discriminator.train_on_batch(imgs, real * (np.random.uniform(0.7, 1.2)))
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake + (np.random.uniform(0.0, 0.3)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # We don't want the discriminator to get updated while the generator is being trained
                self.discriminator.trainable = False

                # Train the generator
                g_loss = self.gan.train_on_batch(noise, real)

                #             if i % 100 == 0:
                #           print ("EPOCH:", epoch, "D LOSS", d_loss, "G LOSS:", g_loss)

                d_loss_acc += d_loss
                g_loss_acc += g_loss

            # Print samples
            if (epoch + 1) % sample_interval == 0:
                self.sample_images(epoch)

            avg_losses['D'].append(d_loss_acc / num_batches)
            avg_losses['G'].append(g_loss_acc / num_batches)

        self.plot_loss(avg_losses)

    def sample_images(self, epoch):
        num_examples = 1
        print('latent_dim: ', self.latent_dim)
        random_noise = np.random.normal(0, 1, size=[num_examples, self.latent_dim])
        print(len(random_noise), ' -random noise:', random_noise)
        for i in random_noise:
            print('len i: ', len(i))
            print(i)

        generated_images = self.generator.predict(random_noise)
        print('generated images 1:', generated_images)
        generated_images = generated_images.reshape(num_examples, 28, 28) * 127.5 + 1
        print('generated images 2:', generated_images)
        plt.figure(figsize=(10, 10))
        for i in range(generated_images.shape[0]):
            plt.subplot(10, 10, i + 1)
            plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
            plt.axis('off')

        # plt.tight_layout()
        plt.suptitle("Samples from G - Epoch = " + str(epoch + 1))
        plt.show()

    def plot_loss(self, losses):

        plt.figure(figsize=(10, 8))
        plt.plot(losses["D"], label="Discriminator loss")
        plt.plot(losses["G"], label="Generator loss")

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title("Loss History")
        plt.show()


clear_session()
gan = SimpleGAN()
gan.train(X_train, epochs=5, batch_size=128, sample_interval=1)


def sample_images2(gan):
    random_noise = np.random.normal(-1.5, 1, size=[1, gan.latent_dim])
    print(random_noise)
    generated_images = gan.generator.predict(random_noise)
    generated_images = generated_images.reshape(1, 28, 28) * 127.5 + 1
    plt.imshow(generated_images[0], interpolation='nearest', cmap='gray_r')
    plt.axis('off')
    return random_noise


noise = sample_images2(gan)


def sample_new_images(gan, noise, n_examples=5):
    new_random_noise = np.random.normal(-1.5, 1, size=[n_examples, gan.latent_dim])
    for i in range(n_examples):
        random_float = np.random.random_sample()
        new_random_noise[i] = noise
        new_random_noise[i][0] = noise[0][0] + random_float + i
        # eventually change more noise in the list
        print(noise[0][0] + random_float + 0.5)
    generated_images = gan.generator.predict(new_random_noise)
    generated_images = generated_images.reshape(n_examples, 28, 28) * 127.5 + 1
    plt.figure(figsize=(10, 10))
    for i in range(generated_images.shape[0]):
        plt.subplot(5, 5, i + 1)
        plt.title(f'{new_random_noise[i][0]:.2f}')
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')


sample_new_images(gan, noise, n_examples=10)