from flask import Flask, render_template, request, send_file, jsonify
import os
import uuid
from projet_anonymisation_medicale import (
    anonymiser_image,
    anonymiser_video,
    verifier_anonymisation,
    verifier_detecteur
)

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_IMAGE = {'jpg', 'jpeg', 'png'}
ALLOWED_VIDEO = {'mp4', 'avi', 'mov'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

verifier_detecteur()


def allowed_file(filename, types):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in types


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/anonymiser', methods=['POST'])
def anonymiser():
    if 'fichier' not in request.files:
        return jsonify({'erreur': 'Aucun fichier envoyé'}), 400

    fichier = request.files['fichier']
    mode = request.form.get('mode', 'flou')
    verifier = request.form.get('verifier', 'false') == 'true'

    if fichier.filename == '':
        return jsonify({'erreur': 'Nom de fichier vide'}), 400

    ext = fichier.filename.rsplit('.', 1)[1].lower() if '.' in fichier.filename else ''
    type_fichier = None

    if ext in ALLOWED_IMAGE:
        type_fichier = 'image'
    elif ext in ALLOWED_VIDEO:
        type_fichier = 'video'
    else:
        return jsonify({'erreur': f'Extension .{ext} non supportée'}), 400

    # Noms de fichiers uniques
    unique_id = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_input.{ext}")
    output_path = os.path.join(OUTPUT_FOLDER, f"{unique_id}_output.{ext}")

    fichier.save(input_path)

    try:
        if type_fichier == 'image':
            resultat = anonymiser_image(input_path, output_path, mode=mode)

            if verifier:
                verification = verifier_anonymisation(output_path)
                resultat['verification'] = verification

            resultat['output_url'] = f"/telecharger/{unique_id}_output.{ext}"
            resultat['input_url'] = f"/apercu/{unique_id}_input.{ext}"

        else:
            resultat = anonymiser_video(input_path, output_path, mode=mode)
            resultat['output_url'] = f"/telecharger/{unique_id}_output.{ext}"

        return jsonify(resultat)

    except Exception as e:
        return jsonify({'erreur': str(e)}), 500


@app.route('/telecharger/<filename>')
def telecharger(filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(path):
        return "Fichier introuvable", 404
    return send_file(path, as_attachment=True)


@app.route('/apercu/<filename>')
def apercu(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        return "Fichier introuvable", 404
    return send_file(path)


@app.route('/apercu_output/<filename>')
def apercu_output(filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(path):
        return "Fichier introuvable", 404
    return send_file(path)


if __name__ == '__main__':
    # Render utilise le port de l'environnement
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)