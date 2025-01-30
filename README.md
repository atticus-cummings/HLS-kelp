Hi, My name is Atticus Cummings and I'm a mechanical engineering student at UCLA. 

This is the github repository for my summer research project through the NASA Student Airborne Research Program. This script downloads multispectral satellite imagery, classifies kelp canopy using a random forest machine learning classifer, and uses the spectral unmixing technique MESMA (Multiple Endmember Spectral Mixture Analysis) to determine inter-pixel kelp fractional coverage. 

I'm currently in the process for reformatting and refactoring it such that it easy for anyone to use. I will also be adding in-depth comments explaining what each step accomplishes. My current workflow requires the Nvidia CUDA toolkit and a linux subsystem for the random forest and spectral unmixing steps. In the near future I aim to create two separate workflows for CPU-based SKlearn processing and GPU-based CUDA CUPY processing.

In the meantime, feel free to try to replicate/borrow any of the code here. 

I will soon upload the training data so one can easily train their own model.

Overview of important scripts

./download data/ 

- get_hls_data.ipynb - This script downloads relevant HLS imagery and ISS elevation maps from NASA Earth Data to your local computer (Requires a Earth Data account: https://urs.earthdata.nasa.gov/users/new ). 

./process data/Linux/

- create_unclassified_training_data.ipynb - This script formats images into training data. It does not produce classified imagery, and the classified band must be manually created in ArcGIS or a similar platform. 

- Train_RF_cuML_v2.ipynb - This script trains the RF classifier

- linux_kelp_v2.ipynb - This script processes the data, running the classifier and MESMA and saving data to the ./processed_imagery folder

./process data/Windows/

- kmeans_usupervised_classifier.ipynb - This script helps you create classified kelp imagery for model training. It utilizes kmeans 
   classification to bin similar pixels, which can then be manually sorted through and assigned a classification. This approach produces impefect training data. 

- Kelp_classify_sklearn_v2.ipynb - Trains SKlearn Random Forest model, capable of running on system without CUDA toolkit.

HLS is based on the Sentinel-2 Tiling system. The scripts are set up to save files according to their assigned tile. If you wish to download images based on the tile ID, you must download the KML file here: https://hls.gsfc.nasa.gov/products-description/tiling-system/ and place this file in the ./maps folder. The file download script will use the coordinates specified in this KML to create an appropriate bounding box.  



