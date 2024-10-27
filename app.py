from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image, ImageDraw
import io
import os
import cv2
import dlib  # Biblioteca para detección de puntos faciales
import base64
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
import pickle

app = Flask(__name__)

# Cargar el predictor de forma facial de dlib
predictor_path = "shape_predictor_68_face_landmarks.dat"
try:
    predictor = dlib.shape_predictor(predictor_path)
except RuntimeError as e:
    print(f"Error al cargar el predictor: {e}")
    exit(1)  # Salir del programa si no se puede cargar el predictor

detector = dlib.get_frontal_face_detector()

# Configura la API de Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def get_drive_service():
    creds = None
    # El archivo token.pickle almacena las credenciales del usuario para la sesión
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # Si no hay credenciales válidas disponibles, solicita al usuario que inicie sesión
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Guarda las credenciales para la próxima sesión
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    # Llama a la API de Google Drive
    service = build('drive', 'v3', credentials=creds)
    return service

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No se ha subido ninguna imagen.'}), 400

    file = request.files['image']

    try:
        # Abrir la imagen con PIL y verificar que sea válida
        try:
            image = Image.open(file)
            image.verify()  # Verificar si la imagen es válida
            image = Image.open(file)  # Volver a abrir la imagen para el procesamiento
        except Exception as e:
            print(f"Error al abrir o verificar la imagen: {str(e)}")
            return jsonify({'error': 'Error al abrir la imagen. Asegúrate de que el archivo sea una imagen válida.'}), 400

        # Convertir la imagen a RGB si no está en ese formato
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image)

        # Convertir la imagen a escala de grises para detección de rostros
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Detectar rostros
        faces = detector(gray)

        # Verificar si se detectaron rostros
        if len(faces) == 0:
            return jsonify({'error': 'No se detectaron rostros en la imagen.'}), 400

        # Crear un objeto de dibujo para agregar las "X"
        draw = ImageDraw.Draw(image)

        # Dibujar puntos faciales en la imagen
        for face in faces:
            landmarks = predictor(gray, face)
            puntos_a_dibujar = [
                21, 22, 17, 25, 36, 37, 38, 42, 43, 44, 30, 51, 57, 48, 54
            ]
            for i in puntos_a_dibujar:
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                draw.line((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 0), width=1)
                draw.line((x - 2, y + 2, x + 2, y - 2), fill=(255, 0, 0), width=1)

        # Guardar la imagen con los puntos faciales en un buffer
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        buf.seek(0)
        image_data = buf.read()

        # Devolver la imagen codificada en base64 para mostrar en el navegador
        image_base64 = 'data:image/jpeg;base64,' + base64.b64encode(image_data).decode('utf-8')

        # Enviar la cantidad de rostros detectados y la imagen procesada
        return jsonify({'matrix': f'Rostros detectados: {len(faces)}', 'image': image_base64})
    except Exception as e:
        # Registrar el error y devolverlo en la respuesta
        print(f"Error al procesar la imagen: {str(e)}")
        return jsonify({'error': f'Error al procesar la imagen: {str(e)}'}), 500

@app.route('/upload_to_drive', methods=['POST'])
def upload_to_drive():
    try:
        # Leer la imagen procesada codificada en base64 desde la solicitud
        image_data = request.json.get('image_data')
        if not image_data:
            return jsonify({'error': 'No se proporcionó la imagen para subir.'}), 400

        # Decodificar la imagen de base64 a bytes
        image_bytes = base64.b64decode(image_data)

        # Subir la imagen procesada a Google Drive
        drive_service = get_drive_service()
        file_metadata = {'name': 'imagen_procesada.jpg'}
        media = MediaFileUpload(io.BytesIO(image_bytes), mimetype='image/jpeg')
        uploaded_file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

        # Devolver el ID del archivo subido
        return jsonify({'file_id': uploaded_file.get('id')})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
