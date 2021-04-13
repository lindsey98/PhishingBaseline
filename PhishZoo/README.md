

Step1:

 - Download the code for all baselines: https://drive.google.com/drive/folders/1YpKR_Nye4E11FCbPbePAAJG4UcqkIsfZ?usp=sharing

 - Navigate to PhishZoo

 - Download benign_sample_30k, phish_sample_30k dataset and targetlist_fit folder, put them into same directory

Step2: Run
```
 python phishzoo.py -f phish_sample_30k -m phish -t targetlist_fit
```

Or 
```
 python phishzoo.py -f benign_sample_30k -m benign -t targetlist_fit
```

Step3: 

  Results will be saved in TestResult folder 

  Format of results: 
  ```

    phish_sample_30k/1&1 Ionos+2019-07-28-22`34`40/shot.png  True 1 3 5 10 0:00:00.597232

    phish_sample_30k/1&1 Ionos+2019-07-28-23`07`22/shot.png True 1 3 5 10 0:00:01.900868
  ```

  Each line is one site, first column is path, second column is True or None, True means the screenshot is identified as phishing, 1 3 5 10 shows whether the true target is inside Top1/3/5/10 predicted targets. Last column is total runtime.  

