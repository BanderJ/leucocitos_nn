import os
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from werkzeug.utils import secure_filename

from .utils import allowed_file, load_and_prep_image
from .model import predict

main_bp = Blueprint('main', __name__)

@main_bp.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('No seleccionaste ningún archivo', 'warning')
            return redirect(request.url)

        if allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_folder = current_app.config['UPLOAD_FOLDER']
            os.makedirs(upload_folder, exist_ok=True)
            upload_path = os.path.join(upload_folder, filename)
            file.save(upload_path)

            img_arr = load_and_prep_image(upload_path)
            clase, confianza = predict(img_arr)
            return render_template('predict.html',
                                   filename=filename,
                                   clase=clase,
                                   confianza=confianza)

        flash('Tipo de archivo no permitido', 'danger')
        return redirect(request.url)

    return render_template('base.html')
