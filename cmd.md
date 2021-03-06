# Prepare data
## Prepare new dataset:
(a: use_interesting_apis)
Use -m to define mapping or remove this flag to auto generate it
python3 main.py \
 -r ../../Dataset/cuckoo_old \
 -d data/obn_iapi/data \
 -p data/obn_iapi/pickle \
 -v data/obn_iapi/vocab_all \
 -m data/mapping_benign_malware.json \
 -fr True -fd True \
 -a True -pv True \
prep


## Prepare new dataset without modifying vocab file (for testing):
(When generate data for testing, must define mapping file by enable flag -m)
python3 main.py \
 -r ../../Dataset/cuckoo_none_detect \
 -d data/n_iapi/data \
 -p data/n_iapi/pickle \
 -v data/obn_iapi/vocab_all \
 -m data/mapping_benign_malware.json \
 -fd True \
 -a True \
prep



# Train:
python3 main.py \
 -r ../../Dataset/cuckoo_old_balance \
 -d data/obn_iapi/data \
 -p data/obn_iapi/pickle \
 -v data/obn_iapi/vocab \
 -m data/mapping_benign_malware.json \
 -fp True \
 -a True \
train \
 --lr 0.001 --weight_decay 0.001 --batch_size 512 --k_fold 20

## Or short:
python3 main.py \
 -p data/obn_iapi/pickle \
 -v data/obn_iapi/vocab \
 -m data/mapping_benign_malware.json \
 -fp True \
train \
 --lr 0.001 --weight_decay 0.001 --batch_size 512 --k_fold 20



# Test
## Test on train/test split data:
python3 main.py \
 -d data/obn_iapi/pickle \
 -p data/obn_iapi/pickle \
 -v data/obn_iapi/vocab \
 -m data/mapping_benign_malware.json \
 -fp True \
test \
 -o __save_results/8925_obn_iapi


## Test on new data:
python3 main.py \
 -p data/n_iapi/pickle \
 -v data/obn_iapi/vocab \
 -m data/mapping_benign_malware.json \
 -fp True \
test_data \
 -o __save_results/8925_obn_iapi



-----------------------------------------------------------

python3 main.py \
 -r ../../Dataset/cuckoo_sm \
 -d data/sm_iapi_size_b0m1/data \
 -p data/sm_iapi_size_b0m1/pickle \
 -v data/sm_fapi_size_b0m1/vocab \
 -fr True -fd True \
 -a True \
prep

python3 main.py \
 -r ../../Dataset/cuckoo_sm \
 -d data/sm_iapi_size_b0m1/data \
 -p data/sm_iapi_size_b0m1/pickle \
 -v data/sm_fapi_size_b0m1/vocab \
 -fp True \
 -a True \
train \
 --lr 0.001 --weight_decay 0.001 --batch_size 16 --k_fold 2



python3 main.py \
 -r ../../Dataset/cuckoo_split/classified \
 -d data/cs_fapi_b0m1/data \
 -p data/cs_fapi_b0m1/pickle \
 -v data/cs_fapi_b0m1/vocab \
 -fr True -fd True \
train \
 --lr 0.001 --weight_decay 0.001 --batch_size 128 --k_fold 10


python3 main.py \
 -r ../../Dataset/cuckoo_split/classified \
 -d data/cs_iapi/data \  
 -p data/cs_iapi/pickle \  
 -v data/cs_iapi/vocab \  
 -fp True \
test \
 -o output/2020-02-07_06-26-11 \
 -c output/2020-02-07_06-26-11/config_edGNN_graph_class.json \
 -cp output/2020-02-07_06-26-11/checkpoints/checkpoint__2020-02-07_06-26-11


python3 main.py \
 -r ../../Dataset/cuckoo_split/none \ 
 -d data/csn_fapi_b0m1/data \
 -p data/csn_fapi_b0m1/pickle \
 -v data/cs_fapi_b0m1/vocab \
 -fr True -fd True \
 -a False \
test_data \
 -o output/2020-02-07_07-58-27 \
 -c output/2020-02-07_07-58-27/config_edGNN_graph_class.json \
 -cp output/2020-02-07_07-58-27/checkpoints/checkpoint__2020-02-07_07-58-27











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
