# MI EEG Decoding

This part conducts a complete MI EEG decoding.

To preprocess dataset, please download it at https://www.bbci.de/competition/iv/ , after application. Aftering unzip it, you should get a folder BCICIV_1_mat of files like BCICIV_calib_ds1a.mat 

Install environment of required packages:
```sh 
pip install xxx
``` 

Then run the following to preprecess data (no need to do so if you already have it processed under ./data/MI1/X.npy)
```sh 
python preprocess_MI_1000hz.py
```   

Then run the following to decode
```sh 
python EEGNet_demo.py
```   