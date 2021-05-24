# Phishing baseline
Implementations of two phishing detection and identification baselines

- EMD: Fu, A. Y., Wenyin, L., & Deng, X. (2006). Detecting phishing web pages with visual similarity assessment based on earth mover's distance (EMD). IEEE transactions on dependable and secure computing, 3(4), 301-311. This paper uses Earth Mover Distance to detect the similarity between two webpage screenshots. 

- Phishzoo: Afroz, S., & Greenstadt, R. (2011, September). Phishzoo: Detecting phishing websites by looking at them. In 2011 IEEE fifth international conference on semantic computing (pp. 368-375). IEEE. This work applies SIFT algorithm to quantify the similarity between two webpage screenshots.

- VisualPhishnet: Abdelnabi, S., Krombholz, K., & Fritz, M. (2020, October). VisualPhishNet: Zero-Day Phishing Website Detection by Visual Similarity. In Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security (pp. 1681-1698). This work trains deep learning Siamese model to compare two webpage screenshots.

- StackModel: Li, Y., Yang, Z., Chen, X., Yuan, H., & Liu, W. (2019). A stacking model using URL and HTML features for phishing webpage detection. Future Generation Computer Systems, 94, 27-39.

- URLNet: Le, H., Pham, Q., Sahoo, D., & Hoi, S. C. (2018). URLNet: Learning a URL representation with deep learning for malicious URL detection. arXiv preprint arXiv:1802.03162.

# Requirements
```  
python == 3.6
opencv-python == 3.4.2.17
opencv-contrib-python == 3.4.2.17
tensorflow == 1.13.1
```


# Instructions
## To run EMD
```
cd EMD/ 
python emd.py -f [path_to_data_folder] -m [benign|phish] -t [path_to_targetlist_folder]
```

## To run PhishZoo, go into 
```
cd PhishZoo/
python phishzoo.py -f [path_to_data_folder] -m [benign|phish] -t [path_to_targetlist_folder]
```
## For VisualPhishnet (Fork from https://github.com/S-Abdelnabi/VisualPhishNet.git)
Download pretrained model [here](https://drive.google.com/file/d/1uCQWaOs2zFR1oAqbd7lZh_73N89YUaHy/view?usp=sharing), [Target list embedding](https://drive.google.com/file/d/1_uCJFK-gdinbblIczYEUFmlHa0c0-ALt/view?usp=sharing), [Targetlist labels](https://drive.google.com/file/d/1l29pzF1BI6KGRFGU-1IyfiWaVcC_j2PV/view?usp=sharing), [Targetlist filename list](https://drive.google.com/file/d/1c4h9F1OjSVz8mAR0xUeH-4AzixW_l6j5/view?usp=sharing)
```
cd VisualPhishnet/
python visualphish_manual.py -f [path_to_data_folder] -r [txt_path_to_save_result]
```
## For StackModel
Download pretrained model [here](https://drive.google.com/file/d/1xxKJNrGxkYN6yqka6EvbiQBdR48RczXL/view?usp=sharing)
```
cd StackModel
python test.py -f [path_to_data_folder] -o [directory_to_save_output]
```
## For URLNet (Fork from https://github.com/Antimalweb/URLNet)
Download pretrained model [here](https://drive.google.com/drive/folders/1YmPRppnp9qpD5xwV4wlNSq5MRIESfIkr?usp=sharing)
```
python test.py \

  --model.emb_mode 5 \

  --data.data_dir [path_to_data_folder] \

  --log.checkpoint_dir output_5/checkpoints/model-2430 \

  --log.output_dir [txt_path_to_save_result] \

  --data.word_dict_dir output_5/words_dict.p \

  --data.char_dict_dir output_5/chars_dict.p \

  --data.subword_dict_dir output_5/subwords_dict.p 
```

