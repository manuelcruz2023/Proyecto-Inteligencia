# Detección temprana de somnolencia (CNN + LSTM + Flask)

## Instalación
Requiere Python 3.10 o 3.11 (TensorFlow 2.15).

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate      Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python scripts/smoke_test.py          # verifica que todo el código funciona (sin dataset)
```

## Dataset
Descargue UTA-RLDD y colóquelo en `data/raw/UTA-RLDD/`. Se detectan recursivamente los videos
`0.*` (Alert), `5.*` (Low Vigilance) y `10.*` (Drowsy); la carpeta padre es el ID del sujeto.

## Pipeline
```bash
python scripts/01_extract_frames.py --workers 4     # ~ decenas de minutos
python scripts/02_build_dataset.py
python scripts/03_train.py                          # añada --no-pretrained si no hay internet
python scripts/04_evaluate.py --split test          # métricas + matriz en reports/
```

## Ejecución
```bash
python run.py                       # desarrollo → http://127.0.0.1:5000
python run_production.py            # producción (Waitress) → http://localhost:8000
gunicorn -c gunicorn.conf.py wsgi:app   # producción en Linux
python scripts/05_infer.py --source 0   # inferencia local con ventana OpenCV
```
La cámara del navegador solo funciona en `localhost` o bajo **HTTPS**
(en un servidor remoto use un proxy inverso con certificado: Nginx/Caddy).