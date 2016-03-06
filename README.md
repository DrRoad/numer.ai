# Artificial Intelligence for Financial Markets. NUMERAI

I though I'd share some of the fun (aka scripts) I've been having competing at NUMERAI using R and other open source tools.

The intent is to share a series of scripts of increasing complexity that would allow someone to decent positions on the leaderboard, build on those scripts and, hopefully, to engourage people to share their solutions. Well... maybe not those that would take you to the Top 5 but humain generosity knows no limits :-)  

Below are main details of the setup to make the results reproduseable and because some of the comptutations are intence and may take hours on some machines
- CPU: AMD FX(tm)-8350; 24GB of RAM
- GPU: GeForce GTX 960/PCIe/SSE2; 4 GB of RAM
- Ubuntu 14.04 64 bit desktop 
- R 3.2.3. 
- RStudio 0.99.486
--

1. [XGBoost (eXtreme Gradient Boosting)] (https://github.com/dmlc/xgboost). XGBoost is often used with exellent results in [Kaggle] (https://www.kaggle.com/) competitions. 
2. [H2O] (http://www.h2o.ai/)
3. H20 Ensembles. Heavily influenced by H2O's Erin LeDell tutorial (https://github.com/h2oai/h2o-tutorials/tree/master/tutorials/ensembles-stacking)
4. [CNTK (Computational Network Toolkit)] (https://github.com/Microsoft/CNTK).
  I failed compiling the toolkit myself or installing from the binaries. It's most likely my fault. I'll keep trying or maybe Microsoft will release a Docker image. Google did it with their TensorFlow. I really wanted to give this one a try because of the outstanding performance and precision results of this deep   learning tool from Microsoft. 
