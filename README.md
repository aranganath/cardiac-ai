# Data Science Challenge 2023 - Cardiac Electrophysiology

This repository contains the notebooks and code for the 2023 Data Science Challenge (DSC) at Lawrence Livermore National Laboratory (LLNL).

- Link : https://data-science.llnl.gov/dsc
- Authors : 
    - Mikel Landajuela (landajuelala1@llnl.gov)
    - Cindy Gonzales (gonzales72@llnl.gov)


## Description

### Training
To train the model, use the following command 
```
python train.py --config_path=configs/config.yaml --save_dir=<save_dir> --tag=<tag>
```

If you do not prefer your own ```<save_dir>```, the program will create one for you in the ```/tmp/``` folder.

If you'd rather submit a slurm script, run the following command
```
sbatch scripts/sbatch_train.sh -c configs/config.yaml -d <save_dir> -t TAG
```

Just make sure, when you are using the slurm batch script, to run it on the right group with the correct access-rights.




### Evaluation
To evaluate the model, use the following command

```
    python eval.py --save_dir=configs/configs.yaml
```


You do not have to specify the ```configs.yaml``` file. The folder already has one.

### How to add your own model+dataset ?
To add your own model, you would need to have a dataset file and a model file.

Once you have the dataset file, place it under ```<dataset>``` folder.

You may keep the model file under ```<models>``` folder. 

For the model file, make sure the ```forward``` is replaced with ```_forward``` and ```nn.Module``` is replaced with ```BaseModel```. Add the line ```(from base import BaseModel)``` to the preamble.

Then, in ```builder.py```, you need to add the model 
```
from datasets import <File-Name>
from models import <Model-FileName>
...
DATASETS ={
    ...
    'yourdata': <File-Name>.<Class-Name>
    ...
}

from models import ..., YourModelFileName
MODELS = {
    ...
    'yourmodel':<Model-FileName>.<Class-Name>
}    
```
to the ```MODEL``` dictionary, and add the dataset to the ```DATASETS``` dictionary.


## Contact
Please contact Aditya Ranganath ([ranganath2@llnl.gov](mailto:ranganath2@llnl.gov)) if you have any request.
