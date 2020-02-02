python3 main.py \
 -r /media/tunguyen/DULIEU/MinhTu/cuckoo_old \
 -d data/old__data_mb_no_edge.json \
 -p data/old__pickle_mb_no_edge \
 -v data/old__vocab_no_edge.txt \
 -fr True  -fd True \
prep


python3 main.py \
 -p data/old__pickle_mb_no_edge \
 -v data/old__vocab_no_edge.txt \
 -fp True \
train \
 --lr 0.001 --weight_decay 0.001 --batch_size 64 --k_fold 20


python3 main.py \
 -p data/old__pickle_mb_no_edge \
 -v data/old__vocab_no_edge.txt \
 -fp True \
test \
 -o output/ \
 -cp 

-----


python3 main.py \
 -r /media/tunguyen/DUONG/cuckoo_test \
 -d data/data_mb_HET_test.json \
 -p data/pickle_mb_HET_test \
 -v data/vocab_test.txt \
 -fp True \
train \
 --lr 0.001 --weight_decay 0.001 --batch_size 8 --k_fold 10


python3 main.py \
 -r /media/tunguyen/DUONG/cuckoo_test \
 -d data/data_mb_HET_test.json \
 -p data/pickle_mb_HET_test \
 -v data/vocab_test.txt \
 -fp True \
test \
 -cp checkpoints/00869


python3 main.py \
 -r ../Dataset/cuckoo_sm \
 -d data/data_mb_HET_full.json \
 -p data/pickle_mb_HET_full \
 -v data/vocab_full.txt \
 -fd True \
train \
 --lr 0.001 --weight_decay 0.001 --batch_size 8 --k_fold 10

python3 main.py \
 -r ../Dataset/cuckoo_all \
 -d data/data_mb_no_edge.json \
 -p data/pickle_mb_no_edge \
 -v data/vocab_no_edge.txt \
 -fr True  -fd True \
prep
