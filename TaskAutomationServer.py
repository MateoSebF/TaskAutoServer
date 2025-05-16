import logging
from flask import Flask, request, jsonify # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
import base64
from TextDetector import TextDetector
from LLMIntegration import LLMIntegration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskAutomationServer:
    def __init__(self):
        self.detector = TextDetector(
            model_path='out/rico/unet_binary/model/4_base_gray_canny(8000).keras',
            tolerance=3,
            min_box_area=25,
            canny_threshold=(5, 20)
        )
        self.llm_integration = LLMIntegration()

    def resolve_petition(self, action: str, image: np.ndarray):
        logger.info(f"Resolviendo petición: {action}")

        cuts = self.detector.detect_and_read(image, advance_clustering=True)
        if not cuts:
            logger.warning("No se encontraron textos en la imagen.")
            return None

        boxes = [box for box, _ in cuts]
        texts = [text for _, text in cuts]
        logger.info(f"Textos encontrados: {texts}")

        response = self.llm_integration.resolve_task(action=action, elements=texts)

        if not isinstance(response, int) or response < 1 or response > len(boxes):
            logger.error("Respuesta inválida del modelo LLM.")
            return None

        selected_box = boxes[response - 1]
        x1, y1, x2, y2 = selected_box
        x = (x1 + x2) // 2
        y = (y1 + y2) // 2
        return (x, y)


# ====== Servidor Flask ======
app = Flask(__name__)
server = TaskAutomationServer()

@app.route('/predict', methods=['POST'])
def predict_json():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Cuerpo JSON requerido'}), 400

    action = data.get('action')
    image_b64 = data.get('image')

    if not action or not image_b64:
        return jsonify({'error': 'Faltan campos: se requiere "action" y "image"'}), 400

    try:
        image_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': f'Error al decodificar la imagen: {str(e)}'}), 400

    if image is None:
        return jsonify({'error': 'La imagen no se pudo decodificar correctamente'}), 400

    result = server.resolve_petition(action, image)
    if result is None:
        return jsonify({'error': 'No se pudo resolver la petición'}), 500

    x, y = result
    return jsonify({'x': x, 'y': y})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
