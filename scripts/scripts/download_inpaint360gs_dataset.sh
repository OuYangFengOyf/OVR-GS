mkdir -p data
cd data

# https://drive.google.com/drive/folders/1UIOPtSJ638VxqLm4yMEcE9hE5mGBwuHH?usp=sharing   All datasets in this repo
gdown --id 1YLpop12JRbzglJfx0FUFUZ2GLaBfZX_x --output TF-OVOR.zip
unzip TF-OVOR.zip

gdown --id 1ev6MFuA_Q49aBW-mNqDdr4IQ7pp4WhZS --output others.zip
unzip others.zip

rm TF-OVOR.zip others.zip

cd ..