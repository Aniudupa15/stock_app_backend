@echo off
REM Weekly ML model retrain - target of the TrainMLModel scheduled task.
"E:\projects\stock_app_backend\.venv\Scripts\python.exe" "E:\projects\stock_app_backend\scripts\train_ml_model.py" >> "E:\projects\stock_app_backend\scripts\train_ml_model.log" 2>&1
