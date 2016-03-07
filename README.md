# Artificial Intelligence for Financial Markets. NUMERAI

I though I'd share some of the fun (aka scripts) I've been having competing at NUMERAI using R and other open source tools.

The intent is to share a series of scripts of different and increasing complexity that would allow someone to be placed in the middle (or maybe higher) positions on the leaderboard, build on those scripts and, hopefully, to engourage people to share their ideas.

Below are the main details of the setup to make the results reproducible and because some of the computations are intense and may take hours on less powerfull machines.
- CPU: AMD FX(tm)-8350; 24GB of RAM
- GPU: GeForce GTX 960/PCIe/SSE2; 4 GB of RAM
- Ubuntu 14.04 64 bit desktop 
- R 3.2.3
- RStudio 0.99.486
 

1. [XGBoost (eXtreme Gradient Boosting)] (https://github.com/dmlc/xgboost). XGBoost is often used with exellent results in [Kaggle] (https://www.kaggle.com/) competitions. 

2. [H2O] (http://www.h2o.ai/). H2O is a popular and powerfull machine learning platform.

3. H20 Ensembles. Heavily influenced by H2O's Erin LeDell [tutorial.] (https://github.com/h2oai/h2o-tutorials/tree/master/tutorials/ensembles-stacking)

4. [CNTK (Computational Network Toolkit)] (https://github.com/Microsoft/CNTK). Deep learning toolkit from Microsoft with outstanding scalability and precision results.
    _Very near future. I failed compiling the toolkit myself or installing from the binaries. It's most likely my fault. I'll keep trying or maybe Microsoft will soon release a newer version or a Docker image._ 

5. [Google TensorFlow] (https://github.com/tensorflow/tensorflow) 

6. 
