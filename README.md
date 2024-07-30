```
Thanks to the pioneering work of the authors of DeepDTA
```
# Resources:

python create_csv.py
```
This returns kiba_train.csv, kiba_test.csv, davis_train.csv, and davis_test.csv, saved in data/ folder. 
```
```
##Train a prediction model
To train a model using training data. The model is chosen if it gains the best MSE for testing data.  
Running 
```

python training.py 0 0 
```
where the first argument is for the index of the datasets, 0/1 for 'davis' or 'kiba', respectively;
 and the third argument is for the index of the cuda, 0/1 for 'cuda:0' or 'cuda:1', respectively. 
```