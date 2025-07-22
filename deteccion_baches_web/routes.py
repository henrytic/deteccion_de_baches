from flask import render_template, request, redirect, url_for, flash, abort
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os

from app import app
from models import db, User, Archivo
from config import UPLOAD_FOLDER, RESULT_FOLDER
from processor import process_image, process_video

# -------- Inicio --------
@app.route('/')
@login_required
def index():
    return render_template('index.html')

# -------- Registro --------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        nombre = request.form['nombre']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        if User.query.filter_by(email=email).first():
            flash('El correo ya está registrado.')
            return redirect(url_for('register'))

        nuevo_usuario = User(nombre=nombre, email=email, password=hashed_password, rol='usuario')
        db.session.add(nuevo_usuario)
        db.session.commit()
        flash('Usuario registrado. Inicia sesión.')
        return redirect(url_for('login'))

    return render_template('register.html')

# -------- Login --------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        usuario = User.query.filter_by(email=email).first()

        if usuario and check_password_hash(usuario.password, password):
            login_user(usuario)
            return redirect(url_for('index'))
        else:
            flash('Credenciales incorrectas.')

    return render_template('login.html')

# -------- Logout --------
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# -------- Subir archivo --------
@app.route('/upload', methods=['POST'])
@login_required
def upload():
    file = request.files['file']
    if not file:
        flash('No se seleccionó ningún archivo.')
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Determinar tipo de archivo y procesar
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        result_path = process_image(filepath)
        tipo = 'imagen'
    elif filename.lower().endswith(('.mp4', '.avi')):
        result_path = process_video(filepath)
        tipo = 'video'
    else:
        flash('Formato de archivo no permitido.')
        return redirect(url_for('index'))

    # Guardar en la base de datos
    nuevo_archivo = Archivo(
        nombre_archivo=filename,
        tipo=tipo,
        resultado_path=result_path,
        user_id=current_user.id
    )
    db.session.add(nuevo_archivo)
    db.session.commit()

    return render_template('result.html', result_path=result_path)

# -------- Panel de administrador --------
@app.route('/admin')
@login_required
def admin_dashboard():
    if current_user.rol != 'admin':
        abort(403)  # Acceso prohibido para usuarios no admin

    archivos = Archivo.query.order_by(Archivo.fecha_hora.desc()).all()
    return render_template('admin_dashboard.html', archivos=archivos)

@app.route('/delete/<int:id>', methods=['POST'])
@login_required
def delete_file(id):
    if current_user.rol != 'admin':
        abort(403)

    archivo = Archivo.query.get_or_404(id)

    # Eliminar el archivo del sistema si existe
    if os.path.exists(archivo.resultado_path):
        os.remove(archivo.resultado_path)

    # Eliminar de la base de datos
    db.session.delete(archivo)
    db.session.commit()

    flash('Archivo eliminado correctamente.')
    return redirect(url_for('admin_dashboard'))
