wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1tfF8gpg9W2C_vi9TmVceLSs9FbcvzjMZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1tfF8gpg9W2C_vi9TmVceLSs9FbcvzjMZ" -O val.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ijTJEpGgz5CCFxtI3ESf3nB_k90ATTvm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ijTJEpGgz5CCFxtI3ESf3nB_k90ATTvm" -O input.zip && rm -rf /tmp/cookies.txt

mkdir data
unzip val.zip && rm val.zip
unzip input.zip && rm input.zip

mv DIV2K_valid_HR data/
mv val_dataset data/