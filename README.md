Ejecutar los siguiente:

py -3 -m venv .venv  
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

Opcional: Si se teine una version diferente al python 3.10 se debe cambiar la variable de entorno

deactivate
Remove-Item -Recurse -Force .venv
py -3.10 -m venv .venv
