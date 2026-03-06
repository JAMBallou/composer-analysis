@echo off
REM python -m src.training.train_temporal src/configs/t1_contrasting.yaml
REM python -m src.training.train_temporal src/configs/t2_similar.yaml
python -m src.training.train_temporal src/configs/t3a_subset.yaml
python -m src.training.train_temporal src/configs/t3b_full.yaml
shutdown -h /s /t 60