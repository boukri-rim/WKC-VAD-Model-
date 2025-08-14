## WKC-VAD-Model
MATLAB implementation of a weighted k-means VAD model using Fisher Discriminant Ratio (FDR) feature selection and automatic online adjustment of class centers for improved detection in noisy environments.
## Contents
[Key features] (## Key features) ,
[Databases ] (## Databases Used),
[Project structure ] (## Project structure) ,
[Code ] (## Code)
## Key features
## Features
- Weighted k-means clustering model for classification using Cityblock distance.
- Extraction of relevant conventional G.729B parameters for VAD using Fisher Discriminant Ratio (FDR).
- Robust performance in noisy environments.
- Fully implemented in MATLAB.
  ## Datasets Used
  This project uses two audio datasets (TIMIT and NOIZEUS) and Aurora noise database ,which are combined to generate noisy audio data for training and evaluation.
  # Speech databases
Tha main audio databases include: TIMIT and NOIZEUS speech datasets which are used in speech processing systems,including speech recongnition,speech enhancement,and voice activity detection.

# TIMIT Acoustic-Phonetic Continuous Corpus
TIMIT is a broadband English speech corpus with about 5 hours of recordings from 630 speakers across 8 American English dialects, each reading 10 phonetically rich sentences, with time-aligned transcriptions for acoustic-phonetic research and ASR development.
The link is: (https://www.kaggle.com/datasets/mfekadu/darpa-timit-acousticphonetic-continuous-speech)
# NOIZEUS 
NOIZEUS is a free noisy speech corpus containing 30 IEEE sentences from 3 male and 3 female speakers, mixed with 8 types of real-world noise (from AURORA) at various SNRs, for benchmarking speech enhancement algorithms.
 The link is: ([https://www.openslr.org/12](https://ecs.utdallas.edu/loizou/speech/noizeus/)).
 ## Project structure
 # Data:
 -Fisher Discriminant ratio coefficients (le rapport discriminant de fisher maximum correctmoy.mat) 
# Model
 - WKC-VAD-model before adjustment (WKC-VAD-Model) 
 - WKC-VAD-model after online adjustment (WKC-VAD-model-update.mat)

# Results
- relative displacement.jpg
- wkc-vad without adjestmwent.fig
- wkc-vad with adjestment.fig
# Scripts
- autocor_ameliore.m
- autocorrelation.m
- build_model_struct.m
- classification_report.m
- critical_parameter.m
- energy_Low_band.m
- FeatureExtractor.m
- fenetrage.m
- fisher_discriminant_ratio.m
- kmeans_improve.m
- kmeans_predict_consistent.m
- pdist2_compat.m
- prepare_feature_weights.m
- smoothing.m
- WKC_VAD_comp_Distance.m
- Zero_crossing_rate.m
- WKC-VAD_Model.m
- update_WKC_VAD_Model
 # Note
 Due to GitHub's file size restrictions, the dataset files and labels.mat are not included in this repository. To obtain these files, please contact me via email [rim.boukri@doc.umc.edu.dz].
