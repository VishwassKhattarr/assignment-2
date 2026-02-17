from src.data_preprocessing import load_no2_data
from src.transformation import transform
from src.train_gan import train_gan
from src.pdf_estimation import estimate_pdf

import numpy as np

# load data
x = load_no2_data("data/india_air_quality.csv")

# transform
z, ar, br = transform(x)

print("ar =", ar)
print("br =", br)

# normalize
z = (z - np.mean(z)) / np.std(z)

# train GAN
G = train_gan(z)

# estimate pdf
estimate_pdf(G)
    