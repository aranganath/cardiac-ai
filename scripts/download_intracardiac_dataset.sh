#!/bin/bash

# Download all the components of the dataset from https://library.ucsd.edu/dc/object/bb29449106
# and put them in the directory intracardiac_dataset

# Make the directory
cd ..
mkdir -p data/ && cd data/
echo "Off-loading data to data/"

# Move into the directory
cd intracardiac_dataset

# Download the first element (README.md)
echo "Downloading the dataset - This may take a while"
wget https://library.ucsd.edu/dc/object/bb29449106/_1_1.md/download -O README.md
for i in {2..22}
do
    wget https://library.ucsd.edu/dc/object/bb29449106/_${i}_1.tgz/download
    echo "Unzipping the file"
    tar -xvzf download
    rm -f download
done
echo "Done"

# Return to the parent directory
cd -