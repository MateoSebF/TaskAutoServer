from typing import List, Tuple, Optional
import logging

import cv2 # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras import Model # type: ignore
from tensorflow.keras.metrics import BinaryAccuracy, MeanIoU # type: ignore
from sklearn.cluster import DBSCAN # type: ignore
import pytesseract # type: ignore

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDetector:
    """
    Detector de texto en imágenes usando un modelo Keras preentrenado.
    Métodos esperan imágenes como np.ndarray en BGR (estilo OpenCV).
    """
    def __init__(
        self,
        model_path: str,
        tolerance: float = 5.0,
        min_box_area: int = 25,
        min_cluster_samples: int = 1,
        canny_threshold: Tuple[int, int] = (5, 20),
        mask_size: Tuple[int, int] = (512, 288),
    ):
        self.model: Model = self._load_model(model_path)
        self.tolerance = tolerance
        self.min_box_area = min_box_area
        self.min_cluster_samples = min_cluster_samples
        self.mask_h, self.mask_w = mask_size
        self.canny_threshold = canny_threshold
        logger.info(
            f"Modelo cargado desde {model_path} | tol={tolerance} | min_area={min_box_area}"
        )

    def _load_model(self, model_path: str) -> Model:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[BinaryAccuracy(), MeanIoU(num_classes=2)]
        )
        return model

    def prepare(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Prepara la imagen para inferencia:
        - redimensiona a mask_size
        - convierte a gris
        - aplica Canny y combina bordes
        - normaliza y expande canal
        """
        if img_bgr is None:
            logger.error("La imagen de entrada es None")
            return None

        # Redimensionar
        img_bgr = cv2.resize(img_bgr, (self.mask_w, self.mask_h), interpolation=cv2.INTER_AREA)

        # Gris y normalización
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Detalle de bordes
        blurred = cv2.GaussianBlur((gray * 255).astype(np.uint8), (5, 5), 1.4)
        edges = cv2.Canny(blurred, *self.canny_threshold)

        # Combina gris y bordes (bordes a 0)
        combined = gray.copy()
        combined[edges > 0] = 0.0
        return combined[..., np.newaxis]

    def predict_mask(self, prepared: np.ndarray) -> np.ndarray:
        """Inferencia y umbral en 0.5 para obtener máscara binaria"""
        batch = prepared[np.newaxis, ...]
        pred = self.model.predict(batch)[0, ..., 0]
        return (pred >= 0.5).astype(np.uint8)

    def get_bounding_boxes(
        self,
        binary_mask: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        coords = np.column_stack(np.where(binary_mask == 1))
        if coords.size == 0:
            return []

        clustering = DBSCAN(eps=self.tolerance, min_samples=self.min_cluster_samples)
        labels = clustering.fit_predict(coords)
        boxes: List[Tuple[int, int, int, int]] = []

        for lbl in set(labels):
            if lbl == -1:
                continue
            pts = coords[labels == lbl]
            ys, xs = pts[:, 0], pts[:, 1]
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            w, h = x_max - x_min + 1, y_max - y_min + 1
            if w * h >= self.min_box_area:
                boxes.append((x_min, y_min, w, h))
        return boxes

    def draw_boxes(
        self,
        original_image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]]
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """Dibuja recuadros escalándolos al tamaño original"""
        img = original_image.copy()
        orig_h, orig_w = img.shape[:2]
        sx, sy = orig_w / self.mask_w, orig_h / self.mask_h
        scaled_boxes: List[Tuple[int, int, int, int]] = []

        for x, y, w, h in boxes:
            x1, y1 = int(x * sx), int(y * sy)
            x2, y2 = int((x + w) * sx), int((y + h) * sy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            scaled_boxes.append((x1, y1, x2, y2))

        return img, scaled_boxes

    def extract_text(
        self,
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[Tuple[int, int, int, int], str]]:
        """Extrae texto de cada región con OCR"""
        results: List[Tuple[Tuple[int, int, int, int], str]] = []
        for (x1, y1, x2, y2) in boxes:
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            text = pytesseract.image_to_string(roi, config="--psm 6").strip()
            if text:
                results.append(((x1, y1, x2, y2), text))
        return results

    def process(
        self,
        image: np.ndarray,
        advance_clustering: bool = False
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[Tuple[int, int, int, int]]
    ]:
        """
        Flujo completo: preparación, predicción, clustering y dibujo.
        Retorna: (original, mask_binary, boxed_image, scaled_boxes)
        """
        prepared = self.prepare(image)
        if prepared is None:
            raise ValueError("Error al preparar la imagen")

        mask = self.predict_mask(prepared)
        if advance_clustering:
            boxes = self.get_bounding_boxes(mask)
        else:
            boxes = self.get_bounding_boxes(mask)

        boxed, scaled = self.draw_boxes(image, boxes)
        return image, mask, boxed, scaled

    def detect_and_read(
        self,
        image: np.ndarray,
        advance_clustering: bool = False
    ) -> List[Tuple[Tuple[int, int, int, int], str]]:
        """
        Proceso completo + OCR en cajas detectadas.
        Retorna lista de (box_coords, texto)
        """
        _, _, boxed, boxes = self.process(image, advance_clustering)
        texts = self.extract_text(boxed, boxes)
        for box, text in texts:
            logger.info(f"Caja {box} -> Texto: {text}")
        return texts
