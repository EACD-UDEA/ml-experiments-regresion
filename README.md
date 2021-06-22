# ml-experiments-template
A template for a Command line interface app for running Machine Learning Experiments


#Instalar dependencias
pip install -r ./service/requirements.txt

#Entrenar el modelo
py .\app.py train .\config.yml

#Evaluar el modelo
py .\app.py eval .\config.yml "2021-06-22 16-42"

#Predecir con el modelo
py .\app.py predict .\config.yml "2021-06-22 14-47"

#Registrar variables (Editor de registro)
#Equipo\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Environment
SERIALIZED_MODEL_PATH=C:\Users\Asus\Documents\Development\WorkSpaces\ecad\machine-learning-ii\demo\models\2021-03-22 02_26_00+00_00\model.joblib
MODEL_LIB_DIR=C:\Users\Asus\Documents\Development\WorkSpaces\ecad\machine-learning-ii\demo\modellling
