# Assignment–2: Learning Probability Density Functions using GAN

## Title
Learning Probability Density Functions using data only using Generative Adversarial Networks (GAN).

---

## Objective
The objective of this project is to learn an unknown probability density function of a transformed random variable using a Generative Adversarial Network (GAN). The model learns the distribution only from samples without assuming any analytical form of the probability density function.

---

## Dataset
India Air Quality Dataset (Kaggle)

Feature used:
- NO2 concentration (x)

Dataset link:
https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data

---

## Methodology

### Step 1: Data Transformation
Each NO2 value is transformed using:

z = x + ar * sin(br * x)

where:

ar = 0.5 × (r mod 7)  
br = 0.3 × (r mod 5 + 1)

r represents the university roll number.

The transformed values (z) are treated as samples from an unknown distribution.

---

### Step 2: PDF Estimation using GAN

A Generative Adversarial Network is trained using only samples of z.

The GAN consists of:

#### Generator
- Takes noise sampled from N(0,1)
- Generates fake samples of z

#### Discriminator
- Distinguishes between real z samples and generated samples
- Outputs probability of authenticity

The generator learns to approximate the underlying distribution of z through adversarial training.

---

### Step 3: PDF Approximation
After training the GAN:

- A large number of samples are generated from the generator.
- Probability density is estimated using histogram density estimation.

---


<img width="340" height="618" alt="image" src="https://github.com/user-attachments/assets/6c03ceef-f13d-4b74-9317-437695a83f2e" />



---

## GAN Architecture

### Generator Network
- Input: 1D Gaussian noise
- Fully connected layers
- ReLU activation
- Output: generated z value

### Discriminator Network
- Input: z sample
- Fully connected layers
- ReLU activation
- Sigmoid output for real/fake classification

---

## Implementation Steps

1. Load NO2 data from dataset.
2. Apply transformation to obtain z.
3. Normalize transformed data.
4. Train GAN using transformed samples.
5. Generate synthetic samples using generator.
6. Estimate and plot probability density.

---

## How to Run

### Install Dependencies

<img width="1011" height="195" alt="image" src="https://github.com/user-attachments/assets/c8a830f7-1a1e-4331-b75a-e259602e3199" />


Output:
- Estimated PDF plot saved in results folder.

---

## Observations

- The generator learns the underlying distribution of the transformed variable from data samples.
- The discriminator helps improve the quality of generated samples through adversarial learning.
- Training stability depends on learning rate and number of epochs.
- Generated distribution approximates the real data distribution.

---

## Requirements
- Python 3.x
- NumPy
- Pandas
- PyTorch
- Matplotlib
- Scikit-learn

## Project Structure

