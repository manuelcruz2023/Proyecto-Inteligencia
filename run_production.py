"""Servidor de producción multiplataforma (Waitress). Windows y Linux."""
from waitress import serve

from app import create_app

if __name__ == "__main__":
    app = create_app()
    print("Servidor en http://0.0.0.0:8000")
    # Un solo proceso: las sesiones viven en memoria. Los hilos atienden peticiones concurrentes.
    serve(app, host="0.0.0.0", port=8000, threads=4)