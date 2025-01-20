# Advanced 3D microstructure generation of solid oxide cell electrodes using conditional generative adversarial network and validation using non-intuitive topological characteristics

This repository presents training a WGAN-gp model for generation of realistic 3D microstructure of SOCs electrodes and structural analysis based on Topological data analysis by applying Persistent homology. 

# Introduction
A generative adversarial network (GAN) model was developed to generate artificial microstructure datasets for SOCs electrodes while controlling volume fraction and specific surface area. Persistent homology analysis further validated the model's ability to capture hidden topological features, demonstrating its effectiveness in generating high-quality microstructures for SOC optimization.
![graphical_abstract](https://github.com/user-attachments/assets/dc82744d-8f85-493b-9bca-e6be257caa09)


# Installation Instructions

To reproduce the analysis and visualizations, install the required software packages:

* torch                       ~2.3.1+cu121
* torchaudio                  ~2.3.1+cu121
* torchvision                 ~0.18.1+cu121
* homcloud                    ~4.4.1
* scikit-learn                ~1.4.2
* scipy                       ~1.13.0
* pandas                      ~2.2.2
* numpy                       ~1.26.4
* matplotlib                  ~3.9.0
  
