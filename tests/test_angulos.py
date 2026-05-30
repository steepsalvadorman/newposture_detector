"""
Tests de extracción de ángulos para perfiles izquierdo y derecho.

Verifica que extraer_angulos_v2 produce valores coherentes con la convención
ROSA independientemente de hacia qué lado mira la persona:

  • Perfil derecho (persona mirando a la derecha, lado izquierdo visible):
    ang_tronco > 0 para inclinación hacia adelante
  • Perfil izquierdo (persona mirando a la izquierda, lado derecho visible):
    ang_tronco > 0 para la MISMA postura (era negativo sin el fix)

Geometría del frame (coordenadas imagen MediaPipe):
  x: 0 = izquierda, 1 = derecha
  y: 0 = arriba,    1 = abajo   (y crece hacia abajo)
"""
import math
import pytest
import numpy as np

from rosa.core.calculos import (
    extraer_angulos_v2,
    calcular_desviacion_vertical,
    calcular_angulo,
    POSE_MIN_VISIBILIDAD,
)


W, H = 640, 480


# ─────────────────────────────────────────────────────────────
# Mock de landmark compatible con calculos.py
# ─────────────────────────────────────────────────────────────

class _LM:
    """Índices MediaPipe Pose usados en calculos.py."""
    NOSE            =  0
    LEFT_SHOULDER   = 11
    RIGHT_SHOULDER  = 12
    LEFT_ELBOW      = 13
    RIGHT_ELBOW     = 14
    LEFT_WRIST      = 15
    RIGHT_WRIST     = 16
    LEFT_HIP        = 23
    RIGHT_HIP       = 24
    LEFT_KNEE       = 25
    RIGHT_KNEE      = 26
    LEFT_ANKLE      = 27
    RIGHT_ANKLE     = 28


VIS_OK  = 0.90   # landmark visible → usado para ángulos
VIS_LOW = 0.10   # landmark no visible → ignorado (< POSE_MIN_VISIBILIDAD)


class _MockLM:
    def __init__(self, x=0.5, y=0.5, vis=VIS_LOW):
        self.x = x
        self.y = y
        self.visibility = vis
        self.presence   = vis


def _make_lms(overrides, n=33):
    """
    Crea lista de n landmarks con vis=VIS_LOW por defecto.
    overrides = {idx: (x, y, vis)}
    """
    lms = [_MockLM() for _ in range(n)]
    for idx, coords in overrides.items():
        x, y = coords[0], coords[1]
        v = coords[2] if len(coords) > 2 else VIS_OK
        lms[idx] = _MockLM(x, y, v)
    return lms


# ─────────────────────────────────────────────────────────────
# Poses de referencia
# ─────────────────────────────────────────────────────────────
#
# Postura NEUTRA para ROSA: inclinación hacia adelante ~10°
#   incl = 90 + ang_tronco,  neutra = 95-110° → ang_tronco ≈ 5-20°
#
# Perfil DERECHO (lado izquierdo visible, persona mira a la derecha →):
#   "adelante" = hacia la derecha en imagen → dx > 0
#   shoulder_left.x > hip_left.x
#
# Perfil IZQUIERDO (lado derecho visible, persona mira a la izquierda ←):
#   "adelante" = hacia la izquierda en imagen → dx < 0
#   shoulder_right.x < hip_right.x
#   Sin corrección: ang_tronco < 0 → ROSA penaliza → BUG
#   Con corrección: ang_tronco = -neg = pos → correcto

# ── Perfil derecho: lado izquierdo hacia la cámara ───────────
_PERFIL_DERECHO = _make_lms({
    _LM.NOSE:           (0.52, 0.10),          # nariz ligeramente adelante
    # Hombro izquierdo adelante (dx > 0 respecto a la cadera)
    _LM.LEFT_SHOULDER:  (0.55, 0.30),          # x=352, y=144
    _LM.LEFT_HIP:       (0.50, 0.55),          # x=320, y=264  → dx=+32
    _LM.LEFT_ELBOW:     (0.65, 0.45),
    _LM.LEFT_WRIST:     (0.70, 0.55),
    _LM.LEFT_KNEE:      (0.35, 0.72),
    _LM.LEFT_ANKLE:     (0.35, 0.88),
    # Lado derecho invisible
    _LM.RIGHT_SHOULDER: (0.55, 0.30, VIS_LOW),
    _LM.RIGHT_HIP:      (0.50, 0.55, VIS_LOW),
    _LM.RIGHT_ELBOW:    (0.65, 0.45, VIS_LOW),
    _LM.RIGHT_WRIST:    (0.70, 0.55, VIS_LOW),
    _LM.RIGHT_KNEE:     (0.65, 0.72, VIS_LOW),
    _LM.RIGHT_ANKLE:    (0.65, 0.88, VIS_LOW),
})

# ── Perfil izquierdo: lado derecho hacia la cámara ───────────
# Espejo horizontal de la misma postura
_PERFIL_IZQUIERDO = _make_lms({
    _LM.NOSE:           (0.48, 0.10),          # nariz adelante (izquierda)
    # Hombro derecho adelante (dx < 0 respecto a la cadera)
    _LM.RIGHT_SHOULDER: (0.45, 0.30),          # x=288, y=144
    _LM.RIGHT_HIP:      (0.50, 0.55),          # x=320, y=264  → dx=-32
    _LM.RIGHT_ELBOW:    (0.35, 0.45),
    _LM.RIGHT_WRIST:    (0.30, 0.55),
    _LM.RIGHT_KNEE:     (0.65, 0.72),
    _LM.RIGHT_ANKLE:    (0.65, 0.88),
    # Lado izquierdo invisible
    _LM.LEFT_SHOULDER:  (0.45, 0.30, VIS_LOW),
    _LM.LEFT_HIP:       (0.50, 0.55, VIS_LOW),
    _LM.LEFT_ELBOW:     (0.35, 0.45, VIS_LOW),
    _LM.LEFT_WRIST:     (0.30, 0.55, VIS_LOW),
    _LM.LEFT_KNEE:      (0.35, 0.72, VIS_LOW),
    _LM.LEFT_ANKLE:     (0.35, 0.88, VIS_LOW),
})

# ── Postura desfavorable: tronco inclinado hacia atrás ───────
_PERFIL_DERECHO_TRONCO_ATRAS = _make_lms({
    _LM.NOSE:           (0.50, 0.10),
    _LM.LEFT_SHOULDER:  (0.45, 0.30),          # dx < 0 → atrás (para derecho)
    _LM.LEFT_HIP:       (0.50, 0.55),
    _LM.LEFT_ELBOW:     (0.40, 0.45),
    _LM.LEFT_WRIST:     (0.35, 0.55),
    _LM.LEFT_KNEE:      (0.35, 0.72),
    _LM.LEFT_ANKLE:     (0.35, 0.88),
    _LM.RIGHT_SHOULDER: (0.45, 0.30, VIS_LOW),
    _LM.RIGHT_HIP:      (0.50, 0.55, VIS_LOW),
    _LM.RIGHT_ELBOW:    (0.40, 0.45, VIS_LOW),
    _LM.RIGHT_WRIST:    (0.35, 0.55, VIS_LOW),
    _LM.RIGHT_KNEE:     (0.65, 0.72, VIS_LOW),
    _LM.RIGHT_ANKLE:    (0.65, 0.88, VIS_LOW),
})

_PERFIL_IZQUIERDO_TRONCO_ATRAS = _make_lms({
    _LM.NOSE:           (0.50, 0.10),
    _LM.RIGHT_SHOULDER: (0.55, 0.30),          # dx > 0 → atrás (para izquierdo)
    _LM.RIGHT_HIP:      (0.50, 0.55),
    _LM.RIGHT_ELBOW:    (0.60, 0.45),
    _LM.RIGHT_WRIST:    (0.65, 0.55),
    _LM.RIGHT_KNEE:     (0.65, 0.72),
    _LM.RIGHT_ANKLE:    (0.65, 0.88),
    _LM.LEFT_SHOULDER:  (0.55, 0.30, VIS_LOW),
    _LM.LEFT_HIP:       (0.50, 0.55, VIS_LOW),
    _LM.LEFT_ELBOW:     (0.60, 0.45, VIS_LOW),
    _LM.LEFT_WRIST:     (0.65, 0.55, VIS_LOW),
    _LM.LEFT_KNEE:      (0.35, 0.72, VIS_LOW),
    _LM.LEFT_ANKLE:     (0.35, 0.88, VIS_LOW),
})


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _angulos(lms):
    """Extrae ángulos y falla el test si devuelve None."""
    result = extraer_angulos_v2(lms, W, H)
    assert result is not None, "extraer_angulos_v2 devolvió None (landmarks insuficientes)"
    return result  # (ang_tronco, ang_rodilla, ang_codo, desv_cuello, ang_muneca)


# ─────────────────────────────────────────────────────────────
# TestAnguloPerfil — comportamiento simétrico izquierdo/derecho
# ─────────────────────────────────────────────────────────────

class TestAnguloPerfil:

    # ── ang_tronco ───────────────────────────────────────────

    def test_tronco_positivo_perfil_derecho(self):
        """Inclinación hacia adelante en perfil derecho → ang_tronco > 0."""
        at, *_ = _angulos(_PERFIL_DERECHO)
        assert at > 0, f"Esperado ang_tronco > 0, obtenido {at:.2f}"

    def test_tronco_positivo_perfil_izquierdo(self):
        """Misma postura en perfil izquierdo → ang_tronco también > 0 (con corrección)."""
        at, *_ = _angulos(_PERFIL_IZQUIERDO)
        assert at > 0, f"Esperado ang_tronco > 0, obtenido {at:.2f}"

    def test_tronco_misma_magnitud_ambos_perfiles(self):
        """Los dos perfiles de la misma postura deben dar el mismo |ang_tronco|."""
        at_d, *_ = _angulos(_PERFIL_DERECHO)
        at_i, *_ = _angulos(_PERFIL_IZQUIERDO)
        assert abs(at_d - at_i) < 0.5, (
            f"Magnitudes distintas: derecho={at_d:.2f}, izquierdo={at_i:.2f}"
        )

    def test_tronco_neutro_en_rango_rosa(self):
        """ang_tronco en postura delantera → inclinación ROSA 95-110°."""
        for lms in (_PERFIL_DERECHO, _PERFIL_IZQUIERDO):
            at, *_ = _angulos(lms)
            incl = 90 + at
            assert 90 < incl < 115, (
                f"incl={incl:.1f} fuera del rango esperado para postura delantera"
            )

    def test_tronco_negativo_o_cero_cuando_inclinado_atras_perfil_derecho(self):
        """Tronco inclinado hacia atrás en perfil derecho → ang_tronco <= 0."""
        at, *_ = _angulos(_PERFIL_DERECHO_TRONCO_ATRAS)
        assert at <= 0, f"Esperado ang_tronco <= 0 para postura atrás, obtenido {at:.2f}"

    def test_tronco_negativo_o_cero_cuando_inclinado_atras_perfil_izquierdo(self):
        """Tronco inclinado hacia atrás en perfil izquierdo → ang_tronco <= 0."""
        at, *_ = _angulos(_PERFIL_IZQUIERDO_TRONCO_ATRAS)
        assert at <= 0, f"Esperado ang_tronco <= 0 para postura atrás, obtenido {at:.2f}"

    # ── ang_rodilla y ang_codo (simétricos, no dependen de dirección) ─

    def test_rodilla_igual_ambos_perfiles(self):
        """La rodilla es un ángulo geométrico simétrico."""
        _, ar_d, *_ = _angulos(_PERFIL_DERECHO)
        _, ar_i, *_ = _angulos(_PERFIL_IZQUIERDO)
        assert abs(ar_d - ar_i) < 1.0, (
            f"rodilla derecho={ar_d:.1f}, izquierdo={ar_i:.1f}"
        )

    def test_codo_igual_ambos_perfiles(self):
        """El codo es un ángulo geométrico simétrico."""
        _, _, ac_d, *_ = _angulos(_PERFIL_DERECHO)
        _, _, ac_i, *_ = _angulos(_PERFIL_IZQUIERDO)
        assert abs(ac_d - ac_i) < 1.0, (
            f"codo derecho={ac_d:.1f}, izquierdo={ac_i:.1f}"
        )

    def test_rodilla_en_rango_fisico(self):
        """La rodilla debe estar entre 45° y 180° para postura sentada normal."""
        for lms in (_PERFIL_DERECHO, _PERFIL_IZQUIERDO):
            _, ar, *_ = _angulos(lms)
            assert 45 <= ar <= 180, f"ang_rodilla={ar:.1f} fuera de rango"

    def test_codo_en_rango_fisico(self):
        for lms in (_PERFIL_DERECHO, _PERFIL_IZQUIERDO):
            _, _, ac, *_ = _angulos(lms)
            assert 20 <= ac <= 180, f"ang_codo={ac:.1f} fuera de rango"

    # ── desv_cuello (consistencia de signo) ─────────────────

    def test_cuello_misma_magnitud_ambos_perfiles(self):
        """Nariz adelante debe dar mismo |desv_cuello| en ambos perfiles."""
        at_d, ar_d, ac_d, dc_d, am_d = _angulos(_PERFIL_DERECHO)
        at_i, ar_i, ac_i, dc_i, am_i = _angulos(_PERFIL_IZQUIERDO)
        assert abs(abs(dc_d) - abs(dc_i)) < 1.0, (
            f"desv_cuello derecho={dc_d:.1f}, izquierdo={dc_i:.1f}"
        )

    # ── ang_muneca (consistencia de signo) ───────────────────

    def test_muneca_misma_magnitud_ambos_perfiles(self):
        at_d, ar_d, ac_d, dc_d, am_d = _angulos(_PERFIL_DERECHO)
        at_i, ar_i, ac_i, dc_i, am_i = _angulos(_PERFIL_IZQUIERDO)
        assert abs(abs(am_d) - abs(am_i)) < 2.0, (
            f"ang_muneca derecho={am_d:.1f}, izquierdo={am_i:.1f}"
        )

    # ── extraer_angulos_v2 con landmarks insuficientes ───────

    def test_devuelve_none_sin_nariz(self):
        lms = _make_lms({
            _LM.LEFT_SHOULDER: (0.55, 0.30),
            _LM.LEFT_HIP: (0.50, 0.55),
            _LM.LEFT_ELBOW: (0.65, 0.45),
            _LM.LEFT_WRIST: (0.70, 0.55),
            _LM.LEFT_KNEE: (0.35, 0.72),
            _LM.LEFT_ANKLE: (0.35, 0.88),
        })
        # Sin NOSE visible → nose=None → return None
        assert extraer_angulos_v2(lms, W, H) is None

    def test_devuelve_none_sin_hombros(self):
        lms = _make_lms({_LM.NOSE: (0.5, 0.10)})
        # mid_sh = None → return None
        assert extraer_angulos_v2(lms, W, H) is None

    def test_devuelve_tupla_cinco_elementos(self):
        result = extraer_angulos_v2(_PERFIL_DERECHO, W, H)
        assert result is not None
        assert len(result) == 5


# ─────────────────────────────────────────────────────────────
# TestCorrecionDireccion — verifica el bug específico
# ─────────────────────────────────────────────────────────────

class TestCorreccionDireccion:
    """
    Antes del fix, con perfil izquierdo (lado derecho visible) ang_tronco
    era negativo para la inclinación hacia adelante, haciendo que puntuar_A4
    calculara incl < 90° y nunca puntuara como neutro.
    """

    def test_bug_ang_tronco_era_negativo_sin_correccion(self):
        """
        Verifica que, sin corrección, ang_tronco sería negativo para perfil izquierdo.
        Con la corrección aplicada, debe ser positivo.
        """
        at, *_ = _angulos(_PERFIL_IZQUIERDO)
        # El lado derecho tiene shoulder.x=0.45 < hip.x=0.50 → vector dx < 0
        # Sin corrección → at < 0. Con corrección → at > 0
        assert at > 0, (
            "BUG: ang_tronco negativo para perfil izquierdo con inclinación hacia adelante. "
            f"Valor: {at:.2f}"
        )

    def test_puntuar_a4_neutro_con_perfil_izquierdo(self):
        """
        puntuar_A4 debe dar base=1 (neutro) para postura sentada correcta,
        independientemente del lado del perfil.
        """
        from rosa.core.calculos import puntuar_A4
        for lms, nombre in (
            (_PERFIL_DERECHO,   "perfil derecho"),
            (_PERFIL_IZQUIERDO, "perfil izquierdo"),
        ):
            at, *_ = _angulos(lms)
            score = puntuar_A4(at, usa_respaldo=True, apoyo_lumbar_adecuado=True,
                               hombros_encogidos=False, respaldo_no_regulable=False)
            # Con at ≈ +15°, incl = 105° → dentro de 95-110° → base = 1, mod = 0 → score = 1
            assert score == 1, (
                f"puntuar_A4={score} ≠ 1 para {nombre} (ang_tronco={at:.2f})"
            )

    def test_ambos_perfiles_mismo_score_a4(self):
        """La puntuación A4 debe ser idéntica para ambos perfiles de la misma postura."""
        from rosa.core.calculos import puntuar_A4
        at_d, *_ = _angulos(_PERFIL_DERECHO)
        at_i, *_ = _angulos(_PERFIL_IZQUIERDO)
        score_d = puntuar_A4(at_d, usa_respaldo=True, apoyo_lumbar_adecuado=True,
                             hombros_encogidos=False, respaldo_no_regulable=False)
        score_i = puntuar_A4(at_i, usa_respaldo=True, apoyo_lumbar_adecuado=True,
                             hombros_encogidos=False, respaldo_no_regulable=False)
        assert score_d == score_i, (
            f"A4 difiere: derecho={score_d} (at={at_d:.1f}), "
            f"izquierdo={score_i} (at={at_i:.1f})"
        )
