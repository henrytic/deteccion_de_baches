import os

# Clave secreta para sesiones Flask
SECRET_KEY = 'clave_secreta_segura_123'

# Configuración base de datos SQLite
SQLALCHEMY_DATABASE_URI = 'sqlite:///site.db'

# Rutas de subida y salida
UPLOAD_FOLDER = os.path.join('static', 'uploads')
RESULT_FOLDER = os.path.join('static', 'results')
