# TF-OVOR result
# https://drive.google.com/drive/folders/1NgqE9SVL8e9BO4ZvIrRQHAhmGIdf9C6g?usp=sharing

gdown --folder 1NgqE9SVL8e9BO4ZvIrRQHAhmGIdf9C6g --output TF-OVOR_result

cd TF-OVOR_result

for f in *.zip; do
    echo "Extracting $f..."
    unzip -o "$f"
done

rm *.zip
cd ..