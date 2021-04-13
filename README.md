# Phishing baseline
Implementations of two phishing detection and identification baselines

- EMD: Fu, A. Y., Wenyin, L., & Deng, X. (2006). Detecting phishing web pages with visual similarity assessment based on earth mover's distance (EMD). IEEE transactions on dependable and secure computing, 3(4), 301-311. This paper uses Earth Mover Distance to detect the similarity between two webpage screenshots. 

- Phishzoo: Afroz, S., & Greenstadt, R. (2011, September). Phishzoo: Detecting phishing websites by looking at them. In 2011 IEEE fifth international conference on semantic computing (pp. 368-375). IEEE. This work applies SIFT algorithm to quantify the similarity between two webpage screenshots.

- VisualPhishnet: Abdelnabi, S., Krombholz, K., & Fritz, M. (2020, October). VisualPhishNet: Zero-Day Phishing Website Detection by Visual Similarity. In Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security (pp. 1681-1698). This work trains deep learning Siamese model to compare two webpage screenshots.

# Project structure
EMD: directory for EMD 
PhishZoo: directory for PhishZoo
VisualPhishnet: code adapted from https://github.com/S-Abdelnabi/VisualPhishNet.git

# Instructions
To run EMD, simply go into EMD/ 
```
python emd.py -f [path_to_data_folder] -m [benign|phish] -t [path_to_targetlist_folder]
```

Similarly, for PhishZoo, go into PhishZoo/
```
python phishzoo.py -f [path_to_data_folder] -m [benign|phish] -t [path_to_targetlist_folder]
```
For VisualPhishnet, go into VisualPhishnet/
```
python visualphish_manual.py -f [path_to_data_folder] -r [txt_path_to_save_result]
```