# ntu-ADL-2022spring-hw1
## Environment
- Pytorch 1.7.1
- Python 3.8.10
## Preprocessing
```bash
sh preprocess.sh 
```
## Train Intent
```bash
python3.8 train_intent.py
```
## Train Slot Tagging
```bash
python3.8 train_slot.py
```
## Predict Intent Classification
```bash
sh intent_cls.sh ${test_file} ${pred_file}
```
## Predict Slot Tagging
```bash
sh slot_tag.sh ${test_file} ${pred_file}
```