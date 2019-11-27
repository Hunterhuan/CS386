# cs386

## preprocess

Read the dataset and divide into train and valid set.

```
python preprocess.py
```

Then you can get the train.json and val.json, which contain the train and valid dataset path.

```python
import os
import json
with open("train.json","r")as f:
	train_dataset = json.load(f)
for dir_name, filename_list in train_dataset.items():
    # label is dir_name
	for filename in filename_list:
		file_dir = os.path.join("dataset", dir_name, filename)
		print(file_dir)
# for valid set, the code is just same as above.
```

