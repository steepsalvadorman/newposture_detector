"""
Evaluador ROSA — NTP 1173 (INSST, 2022)
Lógica de cálculo y extracción geométrica base.
"""

import os
import numpy as np

try:
    import winsound
    WINSOUND_OK = True
except ImportError:
    WINSOUND_OK = False

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────
INTERVALO_EVAL_SEG = 6
GUARDADO_SEG       = 3
MAX_CAMARAS_SCAN   = 6
NIVEL_ACCION       = 5
POSE_MIN_VISIBILIDAD = 0.35
POSE_MIN_PRESENCIA   = 0.30

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_PATH = os.path.join(os.path.expanduser("~"), "pose_landmarker_lite.task")
OBJECT_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
OBJECT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "efficientdet_lite0.tflite")

C = {
    "bg_deep":   "#0A0C10",
    "bg_panel":  "#111318",
    "bg_card":   "#181C22",
    "bg_border": "#252A34",
    "accent":    "#00C8FF",
    "accent2":   "#0087CC",
    "text_hi":   "#F0F4FF",
    "text_mid":  "#9BAEC8",
    "text_lo":   "#56657A",
    "ok":        "#00D68F",
    "warn":      "#F6AD3B",
    "danger":    "#FF4757",
}

NIVELES = {
    1: ("Inapreciable",       "#00D68F"),
    2: ("Bajo",               "#00B87A"),
    3: ("Medio",              "#F6AD3B"),
    4: ("Medio-Alto",         "#E8823A"),
    5: ("ALTO — ACTUAR YA",  "#FF4757"),
}

_LM = {
    "NOSE":            0,
    "LEFT_SHOULDER":  11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW":     13,
    "RIGHT_ELBOW":    14,
    "LEFT_WRIST":     15,
    "RIGHT_WRIST":    16,
    "LEFT_HIP":       23,
    "RIGHT_HIP":      24,
    "LEFT_KNEE":      25,
    "RIGHT_KNEE":     26,
    "LEFT_ANKLE":     27,
    "RIGHT_ANKLE":    28,
}

# ─────────────────────────────────────────────────────────────
# TABLAS NTP 1173
# ─────────────────────────────────────────────────────────────
_TABLA_A = [
    [2,2,3,4,5,6,7,8],
    [2,2,3,4,5,6,7,8],
    [3,3,3,4,5,6,7,8],
    [4,4,4,4,5,6,7,8],
    [5,5,5,5,6,7,8,9],
    [6,6,6,7,7,8,8,9],
    [7,7,7,8,8,9,9,9],
    [8,8,8,9,9,9,9,9],
]
_TABLA_B = [
    [1,1,1,2,3,4,5,6,6],
    [1,1,2,2,3,4,5,6,6],
    [1,2,2,3,3,4,6,7,7],
    [2,2,3,3,4,5,6,8,8],
    [3,3,4,4,5,6,7,8,8],
    [4,4,5,5,6,7,8,9,9],
    [5,5,6,7,8,8,9,9,9],
]
_TABLA_C = [
    [1,1,1,2,3,4,5,6],
    [1,1,2,3,4,5,6,7],
    [1,2,2,3,4,5,6,7],
    [2,3,3,3,5,6,7,8],
    [3,4,4,5,5,6,7,8],
    [4,5,5,6,6,7,8,9],
    [5,6,6,7,7,8,8,9],
    [6,7,7,8,8,9,9,9],
]
_TABLA_D = [
    [1,2,3,4,5,6,7,8,9],
    [2,2,3,4,5,6,7,8,9],
    [3,3,3,4,5,6,7,8,9],
    [4,4,4,4,5,6,7,8,9],
    [5,5,5,5,5,6,7,8,9],
    [6,6,6,6,6,6,7,8,9],
    [7,7,7,7,7,7,7,8,9],
    [8,8,8,8,8,8,8,8,9],
    [9,9,9,9,9,9,9,9,9],
]
_TABLA_E = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9,10],
    [2, 2, 3, 4, 5, 6, 7, 8, 9,10],
    [3, 3, 3, 4, 5, 6, 7, 8, 9,10],
    [4, 4, 4, 4, 5, 6, 7, 8, 9,10],
    [5, 5, 5, 5, 5, 6, 7, 8, 9,10],
    [6, 6, 6, 6, 6, 6, 7, 8, 9,10],
    [7, 7, 7, 7, 7, 7, 7, 8, 9,10],
    [8, 8, 8, 8, 8, 8, 8, 8, 9,10],
    [9, 9, 9, 9, 9, 9, 9, 9, 9,10],
    [10,10,10,10,10,10,10,10,10,10],
]

def _tlu(tabla, fi, ci):
    fi = int(np.clip(fi, 0, len(tabla) - 1))
    ci = int(np.clip(ci, 0, len(tabla[0]) - 1))
    return int(tabla[fi][ci])

# ─────────────────────────────────────────────────────────────
# LÓGICA ROSA v5 — TABLAS COMPLETAS
# ─────────────────────────────────────────────────────────────

def factor_tiempo_F(horas_diarias):
    if horas_diarias > 4:   return +1
    elif horas_diarias < 1: return -1
    else:                   return 0


# ── TABLA A1: Altura de la silla / posición de rodilla ──────
def puntuar_A1(ang_rodilla, pie_llega_suelo, altura_regulable, espacio_insuficiente_piernas=False):
    """
    NTP 1173 A1:
      - Rodillas a 90°                    → 1 (neutro)
      - Pie no llega al suelo             → 3
      - Cualquier otra postura            → 2
      - +1 si existe espacio insuficiente para las piernas
      - +1 si la altura NO es regulable
    """
    if not pie_llega_suelo:
        base = 3
    elif int(round(float(ang_rodilla))) == 90:
        base = 1
    else:
        base = 2

    modificador = 0
    if espacio_insuficiente_piernas:
        modificador += 1
    if not altura_regulable:
        modificador += 1
    return int(np.clip(base + modificador, 1, 9))


# ── TABLA A2: Profundidad del asiento ───────────────────────
def puntuar_A2(dist_rodilla_asiento_cm, profundidad_regulable):
    """
    NTP 1173 A2:
      - 8 cm entre borde del asiento y pierna      → 1
      - Menor o mayor a 8 cm                       → 2
      - +1 si la profundidad NO es regulable
    """
    base = 1 if int(round(float(dist_rodilla_asiento_cm))) == 8 else 2
    modificador = 0 if profundidad_regulable else 1
    return int(np.clip(base + modificador, 1, 9))


# ── TABLA A3: Reposabrazos ───────────────────────────────────
def puntuar_A3(tiene_reposabrazos, reposabrazos_ajustable,
               bordes_afilados, demasiado_anchos, no_regulables,
               reposabrazos_altos_bajos=False):
    """
    NTP 1173 A3:
      - Codos a 90° y hombros relajados    → 1
      - Hombros encogidos o codos bajos    → 2
      Modificadores (acumulables):
        +1 bordes afilados o duros
        +1 demasiado anchos
        +1 no regulables
    """
    if not tiene_reposabrazos:
        base = 2
    elif reposabrazos_altos_bajos:
        base = 2
    else:
        base = 1

    mod = 0
    if tiene_reposabrazos and no_regulables:
        mod += 1
    if bordes_afilados:  mod += 1
    if demasiado_anchos: mod += 1
    return int(np.clip(base + mod, 1, 9))


# ── TABLA A4: Respaldo / tronco ──────────────────────────────
def puntuar_A4(ang_tronco, usa_respaldo, apoyo_lumbar_adecuado,
               hombros_encogidos, respaldo_no_regulable):
    """
    NTP 1173 A4:
      Ángulo de inclinación del tronco respecto vertical:
        inclinacion = 90 + ang_tronco   (ang_tronco medido desde la vertical)
        Postura neutra (>95° y <110°)  → 1
        Sin apoyo lumbar o apoyo inadecuado → 2
        Inclinación >110° o <95°       → 2
        No usa el respaldo (separado)  → 2
      Modificadores:
        +1 superficie alta / hombros encogidos
        +1 respaldo no regulable
    """
    incl = 90 + ang_tronco   # ángulo real del tronco con la vertical

    if not usa_respaldo:
        base = 2
    elif not apoyo_lumbar_adecuado:
        base = 2
    elif 95 < incl < 110:
        base = 1
    else:
        base = 2

    mod = 0
    if hombros_encogidos:        mod += 1
    if respaldo_no_regulable:    mod += 1
    return int(np.clip(base + mod, 1, 9))


# ── TABLA B1: Teléfono ───────────────────────────────────────
def puntuar_B1(telefono_alejado=False,
               sujecion_hombro_cuello=False,
               sin_manos_libres=False,
               factor_t=0):
    """
    NTP 1173 B1:
      Base:
        Cuello recto / 1 mano o manos libres  -> 1
        Telefono alejado > 30 cm              -> 2
      Modificadores:
        +2 sujecion con hombro/cuello
        +1 no existe opcion de manos libres
        +1 / -1 tiempo de uso diario
    """
    base = 2 if telefono_alejado else 1
    mod = 0
    if sujecion_hombro_cuello:
        mod += 2
    if sin_manos_libres:
        mod += 1
    return int(np.clip(base + mod + factor_t, 0, 6))


# ── TABLA B2: Pantalla ───────────────────────────────────────
def puntuar_B2_pantalla(desv_cuello_grados,
                         pantalla_dist_ok,       # True si 40–75 cm y altura correcta
                         pantalla_baja,          # True si la pantalla queda por debajo de 30°
                         pantalla_elevada,        # genera extensión de cuello
                         dist_mayor_75,           # distancia > 75 cm
                         giro_cuello_otra_pantalla,
                         sin_portadocumentos,
                         con_reflejos,
                         factor_t=0):
    """
    NTP 1173 B2:
      Postura base (ángulo cuello respecto vertical):
        pantalla en postura neutra      → 1
        pantalla por debajo de 30°      → 2
        pantalla elevada                → 3
      Modificadores:
        +1 giro cuello a otra pantalla
        +1 sin portadocumentos
        +1 reflejos en pantalla
      Factor tiempo (de horas diarias) se aplica también.
    """
    # de la pantalla respecto a la postura neutra
    if pantalla_elevada:
        base = 3
    elif pantalla_baja:
        base = 2
    elif pantalla_dist_ok:
        base = 1
    else:
        base = 2

    mod = 0
    if giro_cuello_otra_pantalla:   mod += 1
    if sin_portadocumentos:         mod += 1
    if con_reflejos:                mod += 1

    return int(np.clip(base + mod + factor_t, 0, 8))


# ── TABLA C1: Ratón ─────────────────────────────────────────
def puntuar_C1_raton(raton_alineado_hombro,
                      agarre_pinza,
                      raton_teclado_diferente_altura,
                      reposamanos_duro,
                      factor_t=0):
    """
    NTP 1173 C1:
      Base:
        Ratón alineado con el hombro     → 1
        Ratón fuera de alcance/no alin.  → 2
      Modificadores:
        +1 agarre en pinza (ratón pequeño)
        +2 ratón y teclado a diferentes alturas
        +1 reposamanos duro o con presión
        +1 / -1 tiempo de uso diario (tabla F)
    """
    base = 1 if raton_alineado_hombro else 2
    mod = 0
    if agarre_pinza:                      mod += 1
    if raton_teclado_diferente_altura:    mod += 2
    if reposamanos_duro:                  mod += 1
    return int(np.clip(base + mod + factor_t, 0, 7))


# ── TABLA C2: Teclado ────────────────────────────────────────
def puntuar_C2_teclado(ang_muneca,
                        desviacion_al_escribir,
                        alcance_sobre_cabeza,
                        teclado_elevado_hombros,
                        sin_soporte_ajustable,
                        factor_t=0):
    """
    NTP 1173 C2:
      Base:
        Muñeca recta y hombros relajados       → 1
        Extensión muñeca > 15°                 → 2
      Modificadores:
        +1 desviación al escribir (extensión muñeca + hombros)
        +1 alcance por encima de la cabeza
        +1 teclado elevado y hombros encogidos
        +1 sin soporte ajustable
        +1 / -1 tiempo de uso diario (tabla F)
    """
    if abs(ang_muneca) > 15:
        base = 2
    else:
        base = 1

    mod = 0
    if desviacion_al_escribir:
        mod += 1
    if alcance_sobre_cabeza:
        mod += 1
    if teclado_elevado_hombros:
        mod += 1
    if sin_soporte_ajustable:
        mod += 1
    return int(np.clip(base + mod + factor_t, 0, 7))


# ── CÁLCULO ROSA COMPLETO v5 ─────────────────────────────────
def calcular_ROSA_completo_v5(
        ang_tronco, ang_rodilla, ang_codo,
        desv_cuello, ang_muneca,
        horas_silla,
        horas_telefono,
        horas_pantalla,
        horas_raton,
        horas_teclado,
        # --- A1 ---
        pie_llega_suelo, altura_regulable, espacio_insuficiente_piernas,
        # --- A2 ---
        dist_rodilla_cm, profundidad_regulable,
        # --- A3 ---
        tiene_reposabrazos, reposabrazos_ajustable,
        bordes_afilados, brazos_anchos, reposabrazos_no_regulables, reposabrazos_altos_bajos,
        # --- A4 ---
        usa_respaldo, apoyo_lumbar_adecuado,
        hombros_encogidos_silla, respaldo_no_regulable,
        # --- B2 ---
        telefono_alejado, sujecion_hombro_cuello, sin_manos_libres,
        pantalla_dist_ok, pantalla_baja, pantalla_elevada,
        dist_pantalla_mayor_75, giro_otra_pantalla,
        sin_portadocumentos, pantalla_reflejos,
        # --- C1 ---
        raton_alineado, agarre_pinza,
        raton_teclado_dif_altura, reposamanos_duro,
        # --- C2 ---
        desviacion_escribir, alcance_sobre_cabeza,
        teclado_elevado_hombros, sin_soporte_teclado,
):
    ft_silla = factor_tiempo_F(horas_silla)
    ft_pantalla = factor_tiempo_F(horas_pantalla)
    ft_raton = factor_tiempo_F(horas_raton)
    ft_teclado = factor_tiempo_F(horas_teclado)

    a1 = puntuar_A1(ang_rodilla, pie_llega_suelo, altura_regulable, espacio_insuficiente_piernas)
    a2 = puntuar_A2(dist_rodilla_cm, profundidad_regulable)
    a3 = puntuar_A3(tiene_reposabrazos, reposabrazos_ajustable,
                    bordes_afilados, brazos_anchos, reposabrazos_no_regulables,
                    reposabrazos_altos_bajos)
    a4 = puntuar_A4(ang_tronco, usa_respaldo, apoyo_lumbar_adecuado,
                    hombros_encogidos_silla, respaldo_no_regulable)

    suma_asiento = a1 + a2
    suma_soporte = a3 + a4
    tabla_A_val  = _tlu(_TABLA_A, suma_asiento - 2, suma_soporte - 2)
    total_silla  = int(np.clip(tabla_A_val + ft_silla, 1, 10))

    # B1 queda fijo en 1. Ya no se configura desde la UI.
    b1 = 1
    b2 = puntuar_B2_pantalla(desv_cuello, pantalla_dist_ok, pantalla_baja, pantalla_elevada,
                              dist_pantalla_mayor_75, giro_otra_pantalla,
                              sin_portadocumentos, pantalla_reflejos, ft_pantalla)
    tabla_B_val = _tlu(_TABLA_B, b1, b2)

    c1 = puntuar_C1_raton(raton_alineado, agarre_pinza,
                           raton_teclado_dif_altura, reposamanos_duro, ft_raton)
    c2 = puntuar_C2_teclado(ang_muneca, desviacion_escribir, alcance_sobre_cabeza,
                             teclado_elevado_hombros,
                             sin_soporte_teclado, ft_teclado)
    tabla_C_val = _tlu(_TABLA_C, c1, c2)

    tabla_D_val = _tlu(_TABLA_D, tabla_B_val - 1, tabla_C_val - 1)
    rosa = max(total_silla, tabla_D_val)

    return {
        "ang_tronco":  round(ang_tronco, 1),
        "ang_rodilla": round(ang_rodilla, 1),
        "ang_codo":    round(ang_codo, 1),
        "desv_cuello": round(desv_cuello, 1),
        "ang_muneca":  round(ang_muneca, 1),
        "A1": a1, "A2": a2, "A3": a3, "A4": a4,
        "suma_asiento": suma_asiento, "suma_soporte": suma_soporte,
        "tabla_A": tabla_A_val,
        "factor_tiempo_silla": ft_silla,
        "factor_tiempo_telefono": 0,
        "factor_tiempo_pantalla": ft_pantalla,
        "factor_tiempo_raton": ft_raton,
        "factor_tiempo_teclado": ft_teclado,
        "total_silla": total_silla,
        "B1": b1, "B2": b2, "tabla_B": tabla_B_val,
        "C1": c1, "C2": c2, "tabla_C": tabla_C_val,
        "tabla_D": tabla_D_val, "rosa": rosa,
    }

# ─────────────────────────────────────────────────────────────
# ÁNGULOS
# ─────────────────────────────────────────────────────────────
def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def calcular_desviacion_vertical(p_superior, p_inferior):
    vec = np.array(p_superior, dtype=float) - np.array(p_inferior, dtype=float)
    if np.linalg.norm(vec) < 1e-6:
        return 0.0
    dx, dy = float(vec[0]), float(vec[1])
    # 0° significa alineado con la vertical. El signo indica dirección de inclinación.
    return float(np.degrees(np.arctan2(dx, -dy)))


def _landmark_ok(lm):
    if lm is None:
        return False
    if not np.isfinite(lm.x) or not np.isfinite(lm.y):
        return False
    visibility = getattr(lm, "visibility", 1.0)
    presence = getattr(lm, "presence", 1.0)
    return visibility >= POSE_MIN_VISIBILIDAD or presence >= POSE_MIN_PRESENCIA


def _landmark_score(lm):
    if lm is None:
        return 0.0
    return float(min(getattr(lm, "visibility", 1.0), getattr(lm, "presence", 1.0)))


def _punto_landmark(landmarks, idx, w, h):
    lm = landmarks[idx]
    if not _landmark_ok(lm):
        return None
    return [lm.x * w, lm.y * h]


def _promedio_puntos(*pts):
    validos = [p for p in pts if p is not None]
    if not validos:
        return None
    arr = np.array(validos, dtype=float)
    return arr.mean(axis=0).tolist()

def extraer_angulos_v2(landmarks, w, h):
    try:
        nose = _punto_landmark(landmarks, _LM["NOSE"], w, h)
        lsh  = _punto_landmark(landmarks, _LM["LEFT_SHOULDER"], w, h)
        rsh  = _punto_landmark(landmarks, _LM["RIGHT_SHOULDER"], w, h)
        lhi  = _punto_landmark(landmarks, _LM["LEFT_HIP"], w, h)
        rhi  = _punto_landmark(landmarks, _LM["RIGHT_HIP"], w, h)

        if nose is None:
            return None

        mid_sh = _promedio_puntos(lsh, rsh)
        mid_hi = _promedio_puntos(lhi, rhi)
        if mid_sh is None or mid_hi is None:
            return None

        lados = {
            "left": {
                "score": np.mean([
                    _landmark_score(landmarks[_LM["LEFT_SHOULDER"]]),
                    _landmark_score(landmarks[_LM["LEFT_ELBOW"]]),
                    _landmark_score(landmarks[_LM["LEFT_WRIST"]]),
                    _landmark_score(landmarks[_LM["LEFT_HIP"]]),
                    _landmark_score(landmarks[_LM["LEFT_KNEE"]]),
                    _landmark_score(landmarks[_LM["LEFT_ANKLE"]]),
                ]),
                "shoulder": lsh,
                "hip": lhi,
                "elbow": _punto_landmark(landmarks, _LM["LEFT_ELBOW"], w, h),
                "wrist": _punto_landmark(landmarks, _LM["LEFT_WRIST"], w, h),
                "knee": _punto_landmark(landmarks, _LM["LEFT_KNEE"], w, h),
                "ankle": _punto_landmark(landmarks, _LM["LEFT_ANKLE"], w, h),
            },
            "right": {
                "score": np.mean([
                    _landmark_score(landmarks[_LM["RIGHT_SHOULDER"]]),
                    _landmark_score(landmarks[_LM["RIGHT_ELBOW"]]),
                    _landmark_score(landmarks[_LM["RIGHT_WRIST"]]),
                    _landmark_score(landmarks[_LM["RIGHT_HIP"]]),
                    _landmark_score(landmarks[_LM["RIGHT_KNEE"]]),
                    _landmark_score(landmarks[_LM["RIGHT_ANKLE"]]),
                ]),
                "shoulder": rsh,
                "hip": rhi,
                "elbow": _punto_landmark(landmarks, _LM["RIGHT_ELBOW"], w, h),
                "wrist": _punto_landmark(landmarks, _LM["RIGHT_WRIST"], w, h),
                "knee": _punto_landmark(landmarks, _LM["RIGHT_KNEE"], w, h),
                "ankle": _punto_landmark(landmarks, _LM["RIGHT_ANKLE"], w, h),
            },
        }
        lado = max(lados.values(), key=lambda item: item["score"])
        if any(lado[k] is None for k in ("shoulder", "hip", "elbow", "wrist", "knee", "ankle")):
            return None

        ang_tronco = calcular_desviacion_vertical(lado["shoulder"], lado["hip"])
        ang_rodilla = calcular_angulo(lado["hip"], lado["knee"], lado["ankle"])
        ang_codo    = calcular_angulo(lado["shoulder"], lado["elbow"], lado["wrist"])
        desv_cuello = calcular_desviacion_vertical(nose, mid_sh)
        ang_muneca  = calcular_angulo(
            lado["elbow"],
            lado["wrist"],
            [lado["wrist"][0] + 100, lado["wrist"][1]],
        ) - 90

        if not (-80 <= ang_tronco <= 80):
            return None
        if not (45 <= ang_rodilla <= 180):
            return None
        if not (20 <= ang_codo <= 180):
            return None
        if not (-70 <= desv_cuello <= 70):
            return None
        if not (-85 <= ang_muneca <= 85):
            return None

        return ang_tronco, ang_rodilla, ang_codo, desv_cuello, ang_muneca
    except Exception as e:
        print(f"[extraer_angulos] error: {e}")
        return None
