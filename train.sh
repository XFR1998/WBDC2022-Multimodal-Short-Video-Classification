cd './src/ALBEF/Pretraining_QQ/'
python pretrain_my.py
cd ../
python main_kfold.py
cd '../MLM_MFM/Pretraining_QQ/'
python pretrain_my.py
cd ../
python main_kfold.py

cd '../MLM_MFM_ITM/'
# pretrain model, the pretrain task are mfm, mlm and itm
python pretrain.py --use_all_pretrain_data True --max_epochs 20 --val_ratio 0.05 --savedmodel_path save
# fineturn model
python main_kfold.py --max_epochs 6 --val_ratio 0.1 --ema True --use_fgm True



cd ../../



