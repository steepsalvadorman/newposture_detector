"""
Tests de detección de objetos y postura de perfil para EvaluadorROSA.

Cubre:
  - _normalizar_categoria        → normalización de nombres de categoría
  - _objetos_logicos             → mapeo categoría → objeto ROSA
  - OBJECT_TARGETS               → cobertura completa del mapa
  - _parsear_detecciones_objetos → parseo de resultados MediaPipe mockeados
  - _es_postura_perfil           → detección de vista lateral por geometría
  - _verificar_perfil            → wrapper de perfil sobre landmarks
  - _inferir_flags_ergonomicos   → inferencia de flags desde pose + objetos
  - _inferir_mouse_desde_pose_objetos → inferencia de mano sobre ratón
"""
import numpy as np
import pytest

from rosa.detection.camara import (
    OBJECT_TARGETS,
    _es_postura_perfil,
    _inferir_flags_ergonomicos,
    _inferir_mouse_desde_pose_objetos,
    _normalizar_categoria,
    _objetos_logicos,
    _parsear_detecciones_objetos,
    _verificar_perfil,
)


# ─────────────────────────────────────────────────────────────
# Mocks de MediaPipe y landmarks
# ─────────────────────────────────────────────────────────────

class _MockLM:
    def __init__(self, x=0.5, y=0.5, vis=0.9):
        self.x = x
        self.y = y
        self.visibility = vis
        self.presence = vis


class _BBox:
    def __init__(self, origin_x, origin_y, width, height):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height


class _Category:
    def __init__(self, name, score=0.85):
        self.category_name = name
        self.score = score


class _Detection:
    def __init__(self, name, score=0.85, bbox=(50, 50, 200, 150)):
        self.categories = [_Category(name, score)]
        self.bounding_box = _BBox(*bbox)


class _DetectionSinCategorias:
    categories = []
    bounding_box = _BBox(0, 0, 10, 10)


class _ObjResult:
    def __init__(self, detections):
        self.detections = detections


W, H = 640, 480  # dimensiones del frame en todos los tests


def _make_lms(overrides=None, n=33, vis=0.9):
    """Crea n landmarks mock; overrides = {idx: (x, y[, vis])}."""
    lms = [_MockLM(0.5, 0.5, vis) for _ in range(n)]
    if overrides:
        for idx, coords in overrides.items():
            x, y = coords[0], coords[1]
            v = coords[2] if len(coords) > 2 else vis
            lms[idx] = _MockLM(x, y, v)
    return lms


# ── Fixtures de poses representativas ────────────────────────

# Perfil: hombros casi alineados verticalmente (separación < 12% de ancho)
_LMS_PERFIL = _make_lms({
    0:  (0.50, 0.12),          # nariz
    7:  (0.49, 0.12),          # oreja izquierda
    8:  (0.51, 0.12),          # oreja derecha
    11: (0.50, 0.30),          # hombro izq — 0.50*640=320 px
    12: (0.52, 0.30),          # hombro der — 0.52*640=332 px  → sep=12 px < 77 px
    13: (0.42, 0.46),
    14: (0.62, 0.46),
    15: (0.38, 0.62),
    16: (0.68, 0.62),
    23: (0.50, 0.56),
    24: (0.52, 0.56),
    25: (0.35, 0.73),
    26: (0.65, 0.73),
    27: (0.35, 0.88),
    28: (0.65, 0.88),
})

# Frontal: hombros muy separados
_LMS_FRONTAL = _make_lms({
    0:  (0.50, 0.12),
    11: (0.30, 0.30),          # 0.30*640=192 px
    12: (0.70, 0.30),          # 0.70*640=448 px  → sep=256 px > 77 px
    13: (0.20, 0.50),
    14: (0.80, 0.50),
    15: (0.12, 0.65),
    16: (0.88, 0.65),
    23: (0.35, 0.56),
    24: (0.65, 0.56),
    25: (0.35, 0.73),
    26: (0.65, 0.73),
    27: (0.35, 0.88),
    28: (0.65, 0.88),
})

# Mano derecha sobre el ratón (frontal, brazo derecho visible)
_MOUSE_BBOX = (460, 340, 510, 390)  # wrist derecha en (480,360) → dentro del bbox

_LMS_MANO_SOBRE_MOUSE = _make_lms({
    11: (0.30, 0.30, 0.9),    # hombro izq
    12: (0.70, 0.30, 0.9),    # hombro der
    13: (0.20, 0.50, 0.05),   # codo izq  (invisible)
    14: (0.80, 0.50, 0.90),   # codo der  (visible)
    15: (0.10, 0.65, 0.05),   # muñeca izq (invisible)
    16: (0.75, 0.75, 0.90),   # muñeca der → pixel (480, 360) dentro del bbox
    18: (0.76, 0.77, 0.90),   # meñique der
    20: (0.77, 0.78, 0.90),   # índice der
    22: (0.76, 0.76, 0.90),   # pulgar der
    23: (0.35, 0.56, 0.9),
    24: (0.65, 0.56, 0.9),
})

# Mano izquierda lejos del ratón (ratón en esquina derecha)
_MOUSE_BBOX_LEJOS = (540, 390, 610, 430)  # muy lejos de la muñeca izq

_LMS_MANO_LEJOS = _make_lms({
    11: (0.30, 0.30, 0.90),
    12: (0.70, 0.30, 0.90),
    13: (0.20, 0.50, 0.95),   # codo izq  (visible)
    14: (0.80, 0.50, 0.05),   # codo der  (invisible)
    15: (0.10, 0.60, 0.95),   # muñeca izq → pixel (64, 288)
    16: (0.90, 0.60, 0.05),   # muñeca der (invisible)
    17: (0.09, 0.62, 0.95),
    19: (0.08, 0.63, 0.95),
    21: (0.08, 0.61, 0.95),
    23: (0.35, 0.56, 0.90),
    24: (0.65, 0.56, 0.90),
})


# ─────────────────────────────────────────────────────────────
# _normalizar_categoria
# ─────────────────────────────────────────────────────────────

class TestNormalizarCategoria:
    def test_mayusculas_a_minusculas(self):
        assert _normalizar_categoria("Keyboard") == "keyboard"

    def test_mayusculas_completas(self):
        assert _normalizar_categoria("MONITOR") == "monitor"

    def test_espacios_externos(self):
        assert _normalizar_categoria("  tv  ") == "tv"

    def test_ya_normalizado(self):
        assert _normalizar_categoria("mouse") == "mouse"

    def test_none_devuelve_vacio(self):
        assert _normalizar_categoria(None) == ""

    def test_cadena_vacia(self):
        assert _normalizar_categoria("") == ""

    def test_tvmonitor(self):
        assert _normalizar_categoria("tvMonitor") == "tvmonitor"


# ─────────────────────────────────────────────────────────────
# _objetos_logicos
# ─────────────────────────────────────────────────────────────

class TestObjetosLogicos:
    @pytest.mark.parametrize("categoria,esperado", [
        ("keyboard",  ("teclado",)),
        ("mouse",     ("mouse",)),
        ("monitor",   ("pantalla",)),
        ("tv",        ("pantalla",)),
        ("tvmonitor", ("pantalla",)),
        ("laptop",    ("pantalla", "teclado")),
    ])
    def test_mapeo_conocido(self, categoria, esperado):
        resultado = tuple(sorted(_objetos_logicos(categoria)))
        assert resultado == tuple(sorted(esperado))

    def test_categoria_desconocida_vacia(self):
        assert _objetos_logicos("chair")  == ()
        assert _objetos_logicos("person") == ()
        assert _objetos_logicos("")       == ()

    def test_capitalizada_normaliza_antes_de_buscar(self):
        assert _objetos_logicos("Keyboard")  == ("teclado",)
        assert _objetos_logicos("MONITOR")   == ("pantalla",)
        assert _objetos_logicos("Mouse")     == ("mouse",)

    def test_laptop_devuelve_pantalla_y_teclado(self):
        resultado = set(_objetos_logicos("laptop"))
        assert "pantalla" in resultado
        assert "teclado"  in resultado

    def test_todos_los_targets_producen_al_menos_un_objeto(self):
        for clave in OBJECT_TARGETS:
            assert len(_objetos_logicos(clave)) > 0, (
                f"OBJECT_TARGETS['{clave}'] no devuelve objetos lógicos"
            )

    def test_cobertura_completa_de_objetos_rosa(self):
        """pantalla, teclado y mouse deben ser alcanzables por alguna categoría."""
        todos = set()
        for clave in OBJECT_TARGETS:
            todos.update(_objetos_logicos(clave))
        assert "pantalla" in todos
        assert "teclado"  in todos
        assert "mouse"    in todos


# ─────────────────────────────────────────────────────────────
# _parsear_detecciones_objetos
# ─────────────────────────────────────────────────────────────

class TestParsearDeteccionesObjetos:
    def test_resultado_vacio(self):
        dets, presentes = _parsear_detecciones_objetos(_ObjResult([]), W, H)
        assert dets == []
        assert presentes == set()

    def test_deteccion_sin_categorias_ignorada(self):
        dets, presentes = _parsear_detecciones_objetos(
            _ObjResult([_DetectionSinCategorias()]), W, H
        )
        assert dets == []

    def test_categoria_desconocida_ignorada(self):
        dets, _ = _parsear_detecciones_objetos(
            _ObjResult([_Detection("chair", 0.9, (10, 10, 200, 150))]), W, H
        )
        assert dets == []

    def test_teclado(self):
        dets, presentes = _parsear_detecciones_objetos(
            _ObjResult([_Detection("keyboard", 0.88, (10, 200, 250, 80))]), W, H
        )
        assert len(dets) == 1
        assert "teclado" in presentes
        assert dets[0]["label"] == "keyboard"
        assert dets[0]["logical"] == ("teclado",)
        assert abs(dets[0]["score"] - 0.88) < 1e-6

    def test_mouse(self):
        _, presentes = _parsear_detecciones_objetos(
            _ObjResult([_Detection("mouse", 0.75, (300, 300, 80, 60))]), W, H
        )
        assert "mouse" in presentes

    @pytest.mark.parametrize("nombre", ["monitor", "tv", "tvmonitor"])
    def test_pantalla_desde_distintas_categorias(self, nombre):
        _, presentes = _parsear_detecciones_objetos(
            _ObjResult([_Detection(nombre, 0.80, (50, 20, 500, 300))]), W, H
        )
        assert "pantalla" in presentes, f"'{nombre}' no produjo 'pantalla'"

    def test_laptop_produce_pantalla_y_teclado(self):
        _, presentes = _parsear_detecciones_objetos(
            _ObjResult([_Detection("laptop", 0.82, (50, 50, 400, 250))]), W, H
        )
        assert "pantalla" in presentes
        assert "teclado"  in presentes

    def test_multiples_objetos_todos_presentes(self):
        dets, presentes = _parsear_detecciones_objetos(_ObjResult([
            _Detection("keyboard", 0.85, (10, 350, 260, 70)),
            _Detection("mouse",    0.78, (300, 360, 80, 50)),
            _Detection("monitor",  0.91, (50, 20, 500, 280)),
        ]), W, H)
        assert len(dets) == 3
        assert presentes == {"teclado", "mouse", "pantalla"}

    def test_bbox_clippeado_dentro_del_frame(self):
        # bbox que desborda los límites del frame
        dets, _ = _parsear_detecciones_objetos(
            _ObjResult([_Detection("keyboard", 0.9, (-30, -20, 800, 700))]), W, H
        )
        assert len(dets) == 1
        x1, y1, x2, y2 = dets[0]["bbox"]
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= W - 1
        assert y2 <= H - 1

    def test_bbox_ancho_cero_omitido(self):
        # width=0 → x2 == x1 → bbox inválido
        dets, _ = _parsear_detecciones_objetos(
            _ObjResult([_Detection("keyboard", 0.9, (100, 100, 0, 80))]), W, H
        )
        assert dets == []

    def test_bbox_alto_cero_omitido(self):
        dets, _ = _parsear_detecciones_objetos(
            _ObjResult([_Detection("mouse", 0.9, (100, 100, 80, 0))]), W, H
        )
        assert dets == []

    def test_score_preservado(self):
        dets, _ = _parsear_detecciones_objetos(
            _ObjResult([_Detection("mouse", 0.634, (50, 50, 80, 60))]), W, H
        )
        assert abs(dets[0]["score"] - 0.634) < 1e-5

    def test_bbox_almacenado_como_x1y1x2y2(self):
        # origin_x=100, origin_y=50, width=200, height=120
        dets, _ = _parsear_detecciones_objetos(
            _ObjResult([_Detection("monitor", 0.9, (100, 50, 200, 120))]), W, H
        )
        x1, y1, x2, y2 = dets[0]["bbox"]
        assert x1 == 100
        assert y1 == 50
        assert x2 == 300  # 100+200
        assert y2 == 170  # 50+120


# ─────────────────────────────────────────────────────────────
# _es_postura_perfil
# ─────────────────────────────────────────────────────────────

class TestEsPosturaPerfil:
    def test_perfil_hombros_casi_alineados(self):
        lsh    = np.array([320.0, 144.0])
        rsh    = np.array([332.0, 144.0])   # sep = 12 px < 0.12*640=77 px
        sh_mid = (lsh + rsh) / 2
        hip_mid = np.array([326.0, 264.0])
        assert _es_postura_perfil(lsh, rsh, sh_mid, hip_mid, W) is True

    def test_frontal_hombros_muy_separados(self):
        lsh    = np.array([192.0, 144.0])
        rsh    = np.array([448.0, 144.0])   # sep = 256 px >> 77 px
        sh_mid = (lsh + rsh) / 2
        hip_mid = np.array([320.0, 264.0])
        assert _es_postura_perfil(lsh, rsh, sh_mid, hip_mid, W) is False

    def test_sin_hombros_devuelve_false(self):
        assert _es_postura_perfil(None, None, None, None, W) is False
        assert _es_postura_perfil(None, np.array([320.0, 144.0]), None, None, W) is False

    def test_perfil_sin_hip_usa_solo_porcentaje_ancho(self):
        lsh    = np.array([320.0, 144.0])
        rsh    = np.array([332.0, 144.0])   # sep = 12 < 77
        sh_mid = (lsh + rsh) / 2
        assert _es_postura_perfil(lsh, rsh, sh_mid, None, W) is True

    def test_frontal_sin_hip(self):
        lsh    = np.array([192.0, 144.0])
        rsh    = np.array([448.0, 144.0])   # sep = 256 > 77
        sh_mid = (lsh + rsh) / 2
        assert _es_postura_perfil(lsh, rsh, sh_mid, None, W) is False


# ─────────────────────────────────────────────────────────────
# _verificar_perfil
# ─────────────────────────────────────────────────────────────

class TestVerificarPerfil:
    def test_perfil_detectado_con_landmarks(self):
        assert _verificar_perfil(_LMS_PERFIL, W, H) is True

    def test_frontal_no_detectado_con_landmarks(self):
        assert _verificar_perfil(_LMS_FRONTAL, W, H) is False

    def test_no_crashea_con_hombros_invisibles(self):
        lms = _make_lms({
            11: (0.50, 0.30, 0.0),   # vis=0 → filtrado → None → False
            12: (0.52, 0.30, 0.0),
        })
        resultado = _verificar_perfil(lms, W, H)
        assert resultado is False

    def test_devuelve_bool(self):
        assert isinstance(_verificar_perfil(_LMS_PERFIL,   W, H), bool)
        assert isinstance(_verificar_perfil(_LMS_FRONTAL,  W, H), bool)


# ─────────────────────────────────────────────────────────────
# _inferir_flags_ergonomicos
# ─────────────────────────────────────────────────────────────

class TestInferirFlagsErgonomicos:
    def test_sin_landmarks_todo_false(self):
        flags, motivos = _inferir_flags_ergonomicos(None, W, H, [], None)
        assert all(v is False for v in flags.values())
        assert motivos == []

    def test_pantalla_elevada(self):
        # nariz a y=0.30 (144 px). Borde superior pantalla en y=0 < 144-24=120 → elevada
        lms = _make_lms({0: (0.5, 0.30)})
        dets = [{"logical": ("pantalla",), "bbox": (50, 0, 450, 200)}]
        flags, motivos = _inferir_flags_ergonomicos(lms, W, H, dets, None)
        assert flags["pantalla_elevada"] is True
        assert "pantalla alta" in motivos

    def test_pantalla_baja(self):
        # nariz a y=0.10 (48 px). Borde superior pantalla en y=200 > 48+24=72 → baja
        lms = _make_lms({0: (0.5, 0.10)})
        dets = [{"logical": ("pantalla",), "bbox": (50, 200, 450, 350)}]
        flags, _ = _inferir_flags_ergonomicos(lms, W, H, dets, None)
        assert flags["pantalla_baja"] is True

    def test_pantalla_distancia_ok(self):
        # nariz a y=144. Borde superior dentro de ±24 px → ok
        nose_y = int(0.30 * H)         # 144
        top_y  = nose_y + 1            # 145, dentro de [120, 168]
        lms  = _make_lms({0: (0.5, 0.30)})
        dets = [{"logical": ("pantalla",), "bbox": (50, top_y, 450, top_y + 200)}]
        flags, _ = _inferir_flags_ergonomicos(lms, W, H, dets, None)
        assert flags["pantalla_dist_ok"] is True

    def test_sin_detecciones_pantalla_todo_false(self):
        flags, _ = _inferir_flags_ergonomicos(_LMS_PERFIL, W, H, [], None)
        assert flags["pantalla_elevada"] is False
        assert flags["pantalla_baja"]    is False
        assert flags["pantalla_dist_ok"] is False

    def test_alcance_sobre_cabeza(self):
        # muñecas por encima de los hombros (y_muñeca < y_hombro - 5%*H)
        # hombros a y=0.50 (240 px), muñecas a y=0.30 (144 px)
        # condición: 144 < 240 - 24 = 216 → True
        lms = _make_lms({
            11: (0.30, 0.50),   # hombro izq
            12: (0.70, 0.50),   # hombro der
            15: (0.30, 0.30),   # muñeca izq  (alto)
            16: (0.70, 0.30),   # muñeca der  (alto)
        })
        flags, motivos = _inferir_flags_ergonomicos(lms, W, H, [], None)
        assert flags["alcance_sobre_cabeza"] is True
        assert "alcance por encima de la cabeza" in motivos

    def test_hombros_encogidos(self):
        # dist cabeza-hombros = 144-134 = 10. torso = 288-144 = 144.
        # 10 < 0.45*144=64.8 → encogidos
        lms = _make_lms({
            0:  (0.50, 0.279),   # nariz   y≈134 px
            11: (0.30, 0.30),    # hombro izq y=144
            12: (0.70, 0.30),    # hombro der y=144
            23: (0.30, 0.60),    # cadera izq y=288
            24: (0.70, 0.60),    # cadera der y=288
        })
        flags, motivos = _inferir_flags_ergonomicos(lms, W, H, [], None)
        assert flags["hombros_encogidos_silla"] is True
        assert "hombros encogidos" in motivos

    def test_hombros_relajados_no_encogidos(self):
        # dist cabeza-hombros grande → no encogidos
        lms = _make_lms({
            0:  (0.50, 0.05),    # nariz muy arriba y=24
            11: (0.30, 0.40),    # hombro izq y=192
            12: (0.70, 0.40),    # hombro der y=192
            23: (0.30, 0.70),    # cadera izq y=336 → torso=144
            24: (0.70, 0.70),
        })                       # dist=192-24=168 > 0.45*144=64.8 → no encogidos
        flags, _ = _inferir_flags_ergonomicos(lms, W, H, [], None)
        assert flags["hombros_encogidos_silla"] is False


# ─────────────────────────────────────────────────────────────
# _inferir_mouse_desde_pose_objetos
# ─────────────────────────────────────────────────────────────

class TestInferirMouseDesdePoseObjetos:
    def test_sin_detecciones(self):
        r = _inferir_mouse_desde_pose_objetos(_LMS_FRONTAL, W, H, [])
        assert r["mano_sobre_mouse"] is False
        assert r["raton_alineado_inferido"] is None

    def test_sin_landmark_mouse_en_detecciones(self):
        dets = [{"logical": ("teclado",), "score": 0.9, "bbox": (200, 300, 400, 380)}]
        r = _inferir_mouse_desde_pose_objetos(_LMS_FRONTAL, W, H, dets)
        assert r["mano_sobre_mouse"] is False

    def test_sin_landmarks_de_pose(self):
        dets = [{"logical": ("mouse",), "score": 0.9, "bbox": _MOUSE_BBOX}]
        r = _inferir_mouse_desde_pose_objetos(None, W, H, dets)
        assert r["mano_sobre_mouse"] is False

    def test_mano_lejos_del_mouse(self):
        # muñeca izq en (64, 288), ratón en x=[540,610] y=[390,430] → dist≈480 px >> tolerancia
        dets = [{"logical": ("mouse",), "score": 0.9, "bbox": _MOUSE_BBOX_LEJOS}]
        r = _inferir_mouse_desde_pose_objetos(_LMS_MANO_LEJOS, W, H, dets)
        assert r["mano_sobre_mouse"] is False

    def test_mano_sobre_mouse(self):
        # muñeca der en pixel (480, 360) → dentro del bbox (460,340,510,390) → dist=0
        dets = [{"logical": ("mouse",), "score": 0.9, "bbox": _MOUSE_BBOX}]
        r = _inferir_mouse_desde_pose_objetos(_LMS_MANO_SOBRE_MOUSE, W, H, dets)
        assert r["mano_sobre_mouse"] is True

    def test_mano_sobre_mouse_produce_alineacion_inferida(self):
        dets = [{"logical": ("mouse",), "score": 0.9, "bbox": _MOUSE_BBOX}]
        r = _inferir_mouse_desde_pose_objetos(_LMS_MANO_SOBRE_MOUSE, W, H, dets)
        assert r["mano_sobre_mouse"] is True
        assert r["raton_alineado_inferido"] is not None
        assert isinstance(r["raton_alineado_inferido"], bool)

    def test_resultado_tiene_campos_esperados(self):
        r = _inferir_mouse_desde_pose_objetos(_LMS_FRONTAL, W, H, [])
        assert "mano_sobre_mouse"       in r
        assert "raton_alineado_inferido" in r
        assert "lado_mouse"             in r

    def test_lado_mouse_informado_cuando_mano_sobre_mouse(self):
        dets = [{"logical": ("mouse",), "score": 0.9, "bbox": _MOUSE_BBOX}]
        r = _inferir_mouse_desde_pose_objetos(_LMS_MANO_SOBRE_MOUSE, W, H, dets)
        assert r["lado_mouse"] is not None
        assert r["lado_mouse"] in ("izquierdo", "derecho")

    def test_multiples_detecciones_mouse_usa_mayor_score(self):
        # Dos bboxes de ratón: uno lejos (score 0.5) y uno encima de la mano (score 0.9)
        dets = [
            {"logical": ("mouse",), "score": 0.50, "bbox": _MOUSE_BBOX_LEJOS},
            {"logical": ("mouse",), "score": 0.90, "bbox": _MOUSE_BBOX},
        ]
        r = _inferir_mouse_desde_pose_objetos(_LMS_MANO_SOBRE_MOUSE, W, H, dets)
        assert r["mano_sobre_mouse"] is True
