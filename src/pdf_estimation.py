import torch
import matplotlib.pyplot as plt
import os

def estimate_pdf(generator, n_samples=10000):

    # results folder auto create
    os.makedirs("results", exist_ok=True)

    # generate samples
    noise = torch.randn(n_samples, 1)
    samples = generator(noise).detach().numpy()

    # plot histogram density
    plt.figure()
    plt.hist(samples, bins=50, density=True)
    plt.title("Estimated PDF from GAN")
    plt.xlabel("z")
    plt.ylabel("Density")

    # save plot (correct path)
    plt.savefig("results/pdf_plot.png")

    # show window disable (avoid hang)
    plt.close()

    print("PDF saved at results/pdf_plot.png")

    return samples
