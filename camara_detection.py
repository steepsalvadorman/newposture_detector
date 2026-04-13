import os
import queue
import threading
import time
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

from tablas_calculos import (
    INTERVALO_EVAL_SEG,
    MODEL_PATH,
    OBJECT_MODEL_PATH,
    calcular_ROSA_completo_v5,
    extraer_angulos_v2,
)


OBJECT_TARGETS = {
    "tv": ("pantalla",),
    "monitor": ("pantalla",),
    "tvmonitor": ("pantalla",),
    "laptop": ("pantalla", "teclado"),
    "keyboard": ("teclado",),
    "mouse": ("mouse",),
}

_POSE_LM = {
    "NOSE": 0,
    "LEFT_EAR": 7,
    "RIGHT_EAR": 8,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
}


def _dibujar_pose_y_overlay(frame, lms, angulos):
    h, w, _ = frame.shape
    conexiones = [
        (11,12),(11,13),(13,15),(12,14),(14,16),
        (11,23),(12,24),(23,24),
        (23,25),(25,27),(24,26),(26,28),
        (0,11),(0,12),
    ]

    for lm in lms:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 200, 255), -1)
    for i, j in conexiones:
        if i < len(lms) and j < len(lms):
            p1 = (int(lms[i].x * w), int(lms[i].y * h))
            p2 = (int(lms[j].x * w), int(lms[j].y * h))
            cv2.line(frame, p1, p2, (255, 255, 0), 2)

    if angulos:
        at, ar, ac, dc, am = angulos

        def color_ang(condicion_ok):
            return (0, 214, 143) if condicion_ok else (60, 80, 255)

        overlay_data = [
            (f"Tronco:  {at:+.1f}°", color_ang(5.0 < at < 20.0), 20),
            (f"Rodilla: {ar:.1f}°", color_ang(int(round(ar)) == 90), 38),
            (f"Codo:    {ac:.1f}°", color_ang(80.0 <= ac <= 100.0), 56),
            (f"Cuello:  {dc:+.1f}°", color_ang(abs(dc) <= 10.0), 74),
            (f"Muneca:  {am:+.1f}°", color_ang(abs(am) <= 15.0), 92),
        ]
        cv2.rectangle(frame, (w - 200, 8), (w - 4, 102), (10, 12, 16), -1)
        cv2.rectangle(frame, (w - 200, 8), (w - 4, 102), (37, 42, 52), 1)
        for texto, color, y in overlay_data:
            cv2.putText(
                frame,
                texto,
                (w - 196, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

    cv2.rectangle(frame, (0, 0), (320, 20), (10, 12, 16), -1)
    cv2.putText(
        frame,
        "ROSA NTP-1173 v5 Monitor",
        (8, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 200, 255),
        1,
        cv2.LINE_AA,
    )


def _normalizar_categoria(nombre):
    return (nombre or "").strip().lower()


def _objetos_logicos(nombre_categoria):
    return OBJECT_TARGETS.get(_normalizar_categoria(nombre_categoria), ())


def _dibujar_objetos(frame, detecciones):
    if not detecciones:
        return

    h, w, _ = frame.shape
    y = h - 56
    presentes = sorted({obj for det in detecciones for obj in det["logical"]})
    texto_estado = "Objetos: " + (", ".join(presentes) if presentes else "ninguno")
    cv2.rectangle(frame, (8, h - 68), (min(w - 8, 360), h - 8), (10, 12, 16), -1)
    cv2.rectangle(frame, (8, h - 68), (min(w - 8, 360), h - 8), (37, 42, 52), 1)
    cv2.putText(frame, texto_estado, (14, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1, cv2.LINE_AA)

    for det in detecciones:
        x1, y1, x2, y2 = det["bbox"]
        etiqueta = det["label"]
        score = det["score"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            frame,
            f"{etiqueta} {score:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 180, 255),
            1,
            cv2.LINE_AA,
        )


def _pt(lms, idx, w, h):
    if idx >= len(lms):
        return None
    lm = lms[idx]
    return np.array([float(lm.x) * w, float(lm.y) * h], dtype=float)


def _avg_pt(*pts):
    valid = [p for p in pts if p is not None]
    if not valid:
        return None
    return np.mean(valid, axis=0)


def _landmark_score(lms, idx):
    if lms is None or idx >= len(lms):
        return 0.0
    lm = lms[idx]
    return float(min(getattr(lm, "visibility", 1.0), getattr(lm, "presence", 1.0)))


def _es_postura_perfil(lsh, rsh, sh_mid, hip_mid, w):
    if lsh is None or rsh is None or sh_mid is None:
        return False

    separacion_hombros = abs(lsh[0] - rsh[0])
    if hip_mid is not None:
        ref = max(abs(hip_mid[1] - sh_mid[1]), 1.0)
        return separacion_hombros <= max(0.12 * w, 0.45 * ref)
    return separacion_hombros <= 0.12 * w


def _flexion_cabeza_perfil(nose, ear):
    if nose is None or ear is None:
        return None

    dx = abs(float(nose[0] - ear[0]))
    dy = float(nose[1] - ear[1])
    return float(np.degrees(np.arctan2(dy, dx + 1e-6)))


def _lado_perfil(lms, w, h):
    candidatos = []
    for nombre, ear_idx, shoulder_idx in (
        ("izquierdo", _POSE_LM["LEFT_EAR"], _POSE_LM["LEFT_SHOULDER"]),
        ("derecho", _POSE_LM["RIGHT_EAR"], _POSE_LM["RIGHT_SHOULDER"]),
    ):
        ear = _pt(lms, ear_idx, w, h)
        shoulder = _pt(lms, shoulder_idx, w, h)
        if ear is None or shoulder is None:
            continue
        candidatos.append(
            (
                float(np.mean([
                    _landmark_score(lms, ear_idx),
                    _landmark_score(lms, shoulder_idx),
                ])),
                nombre,
                ear,
                shoulder,
            )
        )

    if not candidatos:
        return None

    _, nombre, ear, shoulder = max(candidatos, key=lambda item: item[0])
    return {"nombre": nombre, "ear": ear, "shoulder": shoulder}


def _inferir_flags_ergonomicos(lms, w, h, detecciones, angulos):
    inferidas = {
        "pantalla_baja": False,
        "pantalla_elevada": False,
        "pantalla_dist_ok": False,
        "alcance_sobre_cabeza": False,
        "hombros_encogidos_silla": False,
        "teclado_elevado_hombros": False,
        "reposabrazos_altos_bajos": False,
    }
    motivos = []

    if lms is None:
        return inferidas, motivos

    nose = _pt(lms, _POSE_LM["NOSE"], w, h)
    lsh = _pt(lms, _POSE_LM["LEFT_SHOULDER"], w, h)
    rsh = _pt(lms, _POSE_LM["RIGHT_SHOULDER"], w, h)
    lelb = _pt(lms, _POSE_LM["LEFT_ELBOW"], w, h)
    relb = _pt(lms, _POSE_LM["RIGHT_ELBOW"], w, h)
    lwr = _pt(lms, _POSE_LM["LEFT_WRIST"], w, h)
    rwr = _pt(lms, _POSE_LM["RIGHT_WRIST"], w, h)
    lhi = _pt(lms, _POSE_LM["LEFT_HIP"], w, h)
    rhi = _pt(lms, _POSE_LM["RIGHT_HIP"], w, h)

    sh_mid = _avg_pt(lsh, rsh)
    hip_mid = _avg_pt(lhi, rhi)
    elbow_mid = _avg_pt(lelb, relb)
    wrist_mid = _avg_pt(lwr, rwr)
    ang_codo = angulos[2] if angulos and len(angulos) >= 3 else None

    if nose is not None and sh_mid is not None:
        distancia_cabeza_hombros = sh_mid[1] - nose[1]
        if distancia_cabeza_hombros < (0.16 * h):
            inferidas["hombros_encogidos_silla"] = True
            motivos.append("hombros encogidos")

    if inferidas["hombros_encogidos_silla"] or (ang_codo is not None and not (80.0 <= ang_codo <= 100.0)):
        inferidas["reposabrazos_altos_bajos"] = True
        if ang_codo is not None and not (80.0 <= ang_codo <= 100.0):
            motivos.append("codos fuera de 90°")
        elif inferidas["hombros_encogidos_silla"]:
            motivos.append("reposabrazos fuerzan hombros")

    if wrist_mid is not None and sh_mid is not None:
        if wrist_mid[1] < (sh_mid[1] - 0.05 * h):
            inferidas["alcance_sobre_cabeza"] = True
            motivos.append("alcance por encima de la cabeza")

    bbox_pantallas = [d["bbox"] for d in detecciones if "pantalla" in d.get("logical", ())]
    if nose is not None and bbox_pantallas:
        x1, y1, x2, y2 = max(
            bbox_pantallas,
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
        )
        top_y = float(y1)
        if top_y < (nose[1] - 0.05 * h):
            inferidas["pantalla_elevada"] = True
            motivos.append("pantalla alta")
        elif top_y > (nose[1] + 0.05 * h):
            inferidas["pantalla_baja"] = True
            motivos.append("pantalla baja")
        else:
            inferidas["pantalla_dist_ok"] = True

    if _es_postura_perfil(lsh, rsh, sh_mid, hip_mid, w):
        lado_perfil = _lado_perfil(lms, w, h)
        if lado_perfil is not None:
            flexion = _flexion_cabeza_perfil(nose, lado_perfil["ear"])
            if flexion is not None and flexion > 30.0:
                inferidas["pantalla_baja"] = True
                inferidas["pantalla_dist_ok"] = False
                motivos.append("cabeza agachada >30° en perfil")

    hay_teclado = any("teclado" in d.get("logical", ()) for d in detecciones)
    if hay_teclado and inferidas["hombros_encogidos_silla"]:
        inferidas["teclado_elevado_hombros"] = True
        motivos.append("teclado elevado con hombros encogidos")

    return inferidas, motivos


class HiloCamara(threading.Thread):
    _ANGLE_LIMITS = {
        "at": (-70.0, 70.0),
        "ar": (55.0, 180.0),
        "ac": (25.0, 180.0),
        "dc": (-55.0, 55.0),
        "am": (-70.0, 70.0),
    }
    _ANGLE_JUMP = {
        "at": 10.0,
        "ar": 18.0,
        "ac": 18.0,
        "dc": 8.0,
        "am": 10.0,
    }

    def __init__(self, cola_datos, cola_frames, cola_angulos,
                 horas_diarias, cam_idx, config_ergonomica):
        super().__init__(daemon=True)
        self.cola_datos    = cola_datos
        self.cola_frames   = cola_frames
        self.cola_angulos  = cola_angulos
        self.horas_diarias = horas_diarias
        self.cam_idx       = cam_idx
        self.cfg           = config_ergonomica   # dict con todas las vars de la UI
        self.activo        = True
        self._buf = {"tronco":[], "rodilla":[], "codo":[],
                     "cuello":[], "muneca":[]}
        self._ultimo_resultado = None
        self._lock = threading.Lock()
        self._historial_angulos = {
            "at": deque(maxlen=5),
            "ar": deque(maxlen=5),
            "ac": deque(maxlen=5),
            "dc": deque(maxlen=5),
            "am": deque(maxlen=5),
        }
        self._ultimas_detecciones_obj = []
        self._buf_objetos = {"pantalla": 0, "teclado": 0, "mouse": 0, "muestras": 0}
        self._stop_event = threading.Event()
        self._cap = None
        self._landmarker = None
        self._object_detector = None

    def stop(self):
        self.activo = False
        self._stop_event.set()

        cap = self._cap
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def _suavizar_angulos(self, angulos):
        keys = ("at", "ar", "ac", "dc", "am")
        valores = dict(zip(keys, angulos))
        suavizados = {}

        for key, valor in valores.items():
            if not np.isfinite(valor):
                return None

            vmin, vmax = self._ANGLE_LIMITS[key]
            if not (vmin <= valor <= vmax):
                hist = self._historial_angulos[key]
                if not hist:
                    return None
                valor = float(np.median(hist))

            hist = self._historial_angulos[key]
            if hist:
                base = float(np.median(hist))
                if abs(valor - base) > self._ANGLE_JUMP[key]:
                    valor = base + np.sign(valor - base) * self._ANGLE_JUMP[key]

            hist.append(float(valor))
            suavizados[key] = float(np.median(hist))

        return (
            suavizados["at"],
            suavizados["ar"],
            suavizados["ac"],
            suavizados["dc"],
            suavizados["am"],
        )

    def run(self):
        if not os.path.exists(MODEL_PATH):
            self.cola_datos.put({"error":
                f"Modelo no encontrado en:\n{MODEL_PATH}\n"
                "Ejecuta con internet para descargarlo."})
            return

        def resultado_callback(result, output_image, timestamp_ms):
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                with self._lock:
                    self._ultimo_resultado = result.pose_landmarks[0]

        base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=resultado_callback,
        )

        try:
            landmarker = mp_vision.PoseLandmarker.create_from_options(opts)
            self._landmarker = landmarker
        except Exception as e:
            self.cola_datos.put({"error": f"Error cargando modelo:\n{e}"})
            return

        object_detector = None
        if os.path.exists(OBJECT_MODEL_PATH):
            try:
                obj_opts = mp_vision.ObjectDetectorOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=OBJECT_MODEL_PATH),
                    score_threshold=0.25,
                    max_results=10,
                )
                object_detector = mp_vision.ObjectDetector.create_from_options(obj_opts)
                self._object_detector = object_detector
            except Exception as e:
                print(f"[hilo] object detector error: {e}")

        cap = cv2.VideoCapture(self.cam_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.cam_idx)
        if not cap.isOpened():
            self.cola_datos.put({"error": "No se pudo acceder a la cámara."})
            return
        self._cap = cap

        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            last_eval = time.time()
            last_detect = 0.0
            last_object_detect = 0.0
            ts_ms = 0

            while self.activo and not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    if self._stop_event.wait(0.03):
                        break
                    continue

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                now = time.time()
                if now - last_detect >= (1 / 15):
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    ts_ms = max(ts_ms + 1, int(now * 1000))
                    try:
                        landmarker.detect_async(mp_image, ts_ms)
                        last_detect = now
                    except Exception as e:
                        print(f"[hilo] detect_async error: {e}")

                if object_detector is not None and now - last_object_detect >= 0.5:
                    try:
                        mp_image_obj = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        obj_result = object_detector.detect(mp_image_obj)
                        detecciones = []
                        presentes = set()
                        for det in obj_result.detections:
                            if not det.categories:
                                continue
                            categoria = det.categories[0]
                            label = _normalizar_categoria(categoria.category_name)
                            logical = _objetos_logicos(label)
                            if not logical:
                                continue
                            box = det.bounding_box
                            x1 = max(0, int(box.origin_x))
                            y1 = max(0, int(box.origin_y))
                            x2 = min(w - 1, int(box.origin_x + box.width))
                            y2 = min(h - 1, int(box.origin_y + box.height))
                            if x2 <= x1 or y2 <= y1:
                                continue
                            detecciones.append({
                                "label": label,
                                "logical": logical,
                                "score": float(categoria.score),
                                "bbox": (x1, y1, x2, y2),
                            })
                            presentes.update(logical)
                        self._ultimas_detecciones_obj = detecciones
                        self._buf_objetos["muestras"] += 1
                        for obj in presentes:
                            self._buf_objetos[obj] += 1
                        last_object_detect = now
                    except Exception as e:
                        print(f"[hilo] object detect error: {e}")

                with self._lock:
                    lms = self._ultimo_resultado

                if lms is not None:
                    ang = extraer_angulos_v2(lms, w, h)
                    angulos_overlay = None
                    if ang:
                        suavizados = self._suavizar_angulos(ang)
                        if suavizados:
                            at, ar, ac, dc, am = suavizados
                            angulos_overlay = suavizados
                            self._buf["tronco"].append(at)
                            self._buf["rodilla"].append(ar)
                            self._buf["codo"].append(ac)
                            self._buf["cuello"].append(dc)
                            self._buf["muneca"].append(am)

                            if self.cola_angulos.full():
                                try:
                                    self.cola_angulos.get_nowait()
                                except queue.Empty:
                                    pass
                            self.cola_angulos.put({
                                "at": at, "ar": ar, "ac": ac,
                                "dc": dc, "am": am
                            })
                    _dibujar_pose_y_overlay(frame, lms, angulos_overlay)
                _dibujar_objetos(frame, self._ultimas_detecciones_obj)

                if self.cola_frames.full():
                    try:
                        self.cola_frames.get_nowait()
                    except queue.Empty:
                        pass
                self.cola_frames.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if time.time() - last_eval >= INTERVALO_EVAL_SEG:
                    buf = self._buf
                    if all(len(v) > 0 for v in buf.values()):
                        cfg = self.cfg
                        inferidas, motivos_inferidos = _inferir_flags_ergonomicos(
                            lms,
                            w,
                            h,
                            self._ultimas_detecciones_obj,
                            (
                                float(np.median(buf["tronco"])),
                                float(np.median(buf["rodilla"])),
                                float(np.median(buf["codo"])),
                                float(np.median(buf["cuello"])),
                                float(np.median(buf["muneca"])),
                            ),
                        )
                        pantalla_elevada = cfg["pantalla_elevada"] or inferidas["pantalla_elevada"]
                        pantalla_baja = cfg["pantalla_baja"] or inferidas["pantalla_baja"]
                        pantalla_dist_ok = (
                            (cfg["pantalla_dist_ok"] or inferidas["pantalla_dist_ok"])
                            and not pantalla_elevada
                            and not pantalla_baja
                        )
                        hombros_encogidos_silla = cfg["hombros_encogidos_silla"] or inferidas["hombros_encogidos_silla"]
                        reposabrazos_altos_bajos = cfg["reposabrazos_altos_bajos"] or inferidas["reposabrazos_altos_bajos"]
                        alcance_sobre_cabeza = cfg["alcance_sobre_cabeza"] or inferidas["alcance_sobre_cabeza"]
                        teclado_elevado_hombros = cfg["teclado_elevado_hombros"] or inferidas["teclado_elevado_hombros"]
                        resultado = calcular_ROSA_completo_v5(
                            ang_tronco=float(np.median(buf["tronco"])),
                            ang_rodilla=float(np.median(buf["rodilla"])),
                            ang_codo=float(np.median(buf["codo"])),
                            desv_cuello=float(np.median(buf["cuello"])),
                            ang_muneca=float(np.median(buf["muneca"])),
                            horas_silla=cfg["horas_silla"],
                            horas_telefono=cfg["horas_telefono"],
                            horas_pantalla=cfg["horas_pantalla"],
                            horas_raton=cfg["horas_raton"],
                            horas_teclado=cfg["horas_teclado"],
                            pie_llega_suelo=cfg["pie_llega_suelo"],
                            altura_regulable=cfg["altura_regulable"],
                            espacio_insuficiente_piernas=cfg["espacio_insuficiente_piernas"],
                            dist_rodilla_cm=cfg["dist_rodilla_cm"],
                            profundidad_regulable=cfg["profundidad_regulable"],
                            tiene_reposabrazos=cfg["tiene_reposabrazos"],
                            reposabrazos_ajustable=cfg["reposabrazos_ajustable"],
                            bordes_afilados=cfg["bordes_afilados"],
                            brazos_anchos=cfg["brazos_anchos"],
                            reposabrazos_no_regulables=cfg["reposabrazos_no_regulables"],
                            reposabrazos_altos_bajos=reposabrazos_altos_bajos,
                            usa_respaldo=cfg["usa_respaldo"],
                            apoyo_lumbar_adecuado=cfg["apoyo_lumbar_adecuado"],
                            hombros_encogidos_silla=hombros_encogidos_silla,
                            respaldo_no_regulable=cfg["respaldo_no_regulable"],
                            telefono_alejado=cfg["telefono_alejado"],
                            sujecion_hombro_cuello=cfg["sujecion_hombro_cuello"],
                            sin_manos_libres=cfg["sin_manos_libres"],
                            pantalla_dist_ok=pantalla_dist_ok,
                            pantalla_baja=pantalla_baja,
                            pantalla_elevada=pantalla_elevada,
                            dist_pantalla_mayor_75=cfg["dist_pantalla_mayor_75"],
                            giro_otra_pantalla=cfg["giro_otra_pantalla"],
                            sin_portadocumentos=cfg["sin_portadocumentos"],
                            pantalla_reflejos=cfg["pantalla_reflejos"],
                            raton_alineado=cfg["raton_alineado"],
                            agarre_pinza=cfg["agarre_pinza"],
                            raton_teclado_dif_altura=cfg["raton_teclado_dif_altura"],
                            reposamanos_duro=cfg["reposamanos_duro"],
                            desviacion_escribir=cfg["desviacion_escribir"],
                            alcance_sobre_cabeza=alcance_sobre_cabeza,
                            teclado_elevado_hombros=teclado_elevado_hombros,
                            sin_soporte_teclado=cfg["sin_soporte_teclado"],
                        )
                        obj_pantalla = self._buf_objetos["pantalla"] > 0
                        obj_teclado = self._buf_objetos["teclado"] > 0
                        obj_mouse = self._buf_objetos["mouse"] > 0
                        contexto_completo = obj_pantalla and obj_teclado and obj_mouse
                        resultado["obj_pantalla"] = obj_pantalla
                        resultado["obj_teclado"] = obj_teclado
                        resultado["obj_mouse"] = obj_mouse
                        resultado["contexto_completo"] = contexto_completo
                        resultado["contexto_objetos"] = (
                            f"pantalla={'si' if obj_pantalla else 'no'}, "
                            f"teclado={'si' if obj_teclado else 'no'}, "
                            f"mouse={'si' if obj_mouse else 'no'}"
                        )
                        resultado["flags_inferidos"] = inferidas
                        resultado["motivos_inferidos"] = ", ".join(dict.fromkeys(motivos_inferidos))
                        resultado["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                        self.cola_datos.put(resultado)
                        for k in self._buf:
                            self._buf[k].clear()
                        self._buf_objetos = {"pantalla": 0, "teclado": 0, "mouse": 0, "muestras": 0}
                    last_eval = time.time()
        finally:
            self._cap = None
            try:
                cap.release()
            except Exception:
                pass
            try:
                landmarker.close()
            except Exception:
                pass
            self._landmarker = None
            if object_detector is not None:
                try:
                    object_detector.close()
                except Exception:
                    pass
            self._object_detector = None
