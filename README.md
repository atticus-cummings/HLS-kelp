Hi, My name is Atticus Cummings and I'm a mechanical engineering student at UCLA. 

This is the github repository for my summer research project through the NASA Student Airborne Research Program. This script downloads multispectral satellite imagery, classifies kelp canopy using a random forest machine learning classifer, and uses the spectral unmixing technique MESMA (Multiple Endmember Spectral Mixture Analysis) to determine inter-pixel kelp fractional coverage. 

I'm currently in the process for reformatting and refactoring it such that it easy for anyone to use. I will also be adding in-depth comments explaining what each step accomplishes. My current workflow requires the Nvidia CUDA toolkit and a linux subsystem for the random forest and spectral unmixing steps. In the near future I aim to create two separate workflows for CPU-based SKlearn processing and GPU-based CUDA CUPY processing.

In the meantime, feel free to try to replicate/borrow any of the code here. 

I will also soon upload training data so one can easily train their own model.
