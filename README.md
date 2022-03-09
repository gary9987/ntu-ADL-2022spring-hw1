# ntu-ADL-2022spring-hw1
## Environment
- Pytorch 1.7.1
## Preprocessing
```bash
sh preprocess.sh 
```
## Intent Classification
```bash
sh intent_cls.sh ${test_file} ${pred_file}
```
## Slot Tagging
```bash
sh slot_tag.sh ${test_file} ${pred_file}
```