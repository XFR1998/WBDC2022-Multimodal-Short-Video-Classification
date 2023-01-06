cd './src/ALBEF/'
python inference_kfold_avg_logits.py

cd '../MLM_MFM/'
python inference_kfold_avg_logits.py

cd '../MLM_MFM_ITM/'
# inference test_b
python inference_kfold.py --test_annotation '../../data/annotations/test_b.json' --test_zip_feats '../../data/zip_feats/test_b.zip'

cd '../'
python inference_many_kfold_models.py


