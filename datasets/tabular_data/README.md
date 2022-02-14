
## Using Decision Trees to Construct Weak Supervision Tasks on Tabular Data

This script requires packages `openml`, `sklearn`, `pandas`


Run the script to generate dataset:
```
python generate_dataset.py --data_dir ../ --data_name mushroom --n_trees 20 --max_depth 3 --max_features 3
```
Where `n_trees` specify how many decision trees will be used to generate labeling functions, note that the number of final labeling functions could be less than number of trees because we remove trees resulting duplicate labeling;
Note that `n_trees`, `max_depth`, `max_depth`, and `max_features` are arguments for `sklearn.ensemble.RandomForestClassifier`.
This command will use the `mushroom` dataset from `openml`, specifically, the dataset will be downloaded by the following code in the script

```
dataset = openml.datasets.get_dataset(args.data_name)
```

So make sure the `data_name` matches dataset in openml.
Then it will use decision tree to extract labeling functions.
Finally, a new dataset folder `mushroom` will show up:
```
datasets 
     -- mushroom
         |-- train.json
         |-- valid.json
         |-- test.json
         |-- label.json
         |-- rules.json
```

You could check out the generated labeling functions in the file `rules.json`, where each rule contains a list of triplet `(a:int, b:float, c:bool)`. Each triplet describe one condition, `a` is the id of the associated feature, `b` is the threshold for this condition, `c` indicates the direction of the inequality (True means that the condition holds if the feature value less than the threshold)


