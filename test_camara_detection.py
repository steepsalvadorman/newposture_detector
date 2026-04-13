import unittest

from camara_detection import _flexion_cabeza_perfil, _inferir_flags_ergonomicos


class _FakeLandmark:
    def __init__(self, x, y, visibility=1.0, presence=1.0):
        self.x = x
        self.y = y
        self.visibility = visibility
        self.presence = presence


def _landmarks_base():
    lms = [_FakeLandmark(0.0, 0.0, 0.0, 0.0) for _ in range(33)]
    lms[0] = _FakeLandmark(0.52, 0.24)
    lms[7] = _FakeLandmark(0.50, 0.22)
    lms[8] = _FakeLandmark(0.53, 0.22)
    lms[11] = _FakeLandmark(0.48, 0.42)
    lms[12] = _FakeLandmark(0.52, 0.42)
    lms[13] = _FakeLandmark(0.47, 0.54)
    lms[14] = _FakeLandmark(0.53, 0.54)
    lms[15] = _FakeLandmark(0.47, 0.66)
    lms[16] = _FakeLandmark(0.53, 0.66)
    lms[23] = _FakeLandmark(0.48, 0.68)
    lms[24] = _FakeLandmark(0.52, 0.68)
    return lms


class TestCamaraDetection(unittest.TestCase):
    def test_flexion_cabeza_perfil_supera_30_grados(self):
        flexion = _flexion_cabeza_perfil((150.0, 120.0), (100.0, 80.0))
        self.assertGreater(flexion, 30.0)

    def test_infiere_reposabrazos_altos_bajos_por_codo_fuera_de_90(self):
        flags, motivos = _inferir_flags_ergonomicos(
            _landmarks_base(),
            640,
            480,
            [],
            (10.0, 90.0, 118.0, 0.0, 0.0),
        )

        self.assertTrue(flags["reposabrazos_altos_bajos"])
        self.assertIn("codos fuera de 90°", motivos)

    def test_infiere_pantalla_baja_por_flexion_de_cabeza_en_perfil(self):
        lms = _landmarks_base()
        lms[11] = _FakeLandmark(0.495, 0.42)
        lms[12] = _FakeLandmark(0.505, 0.42)
        lms[23] = _FakeLandmark(0.495, 0.68)
        lms[24] = _FakeLandmark(0.505, 0.68)
        lms[7] = _FakeLandmark(0.50, 0.24)
        lms[8] = _FakeLandmark(0.50, 0.24)
        lms[0] = _FakeLandmark(0.60, 0.36)

        flags, motivos = _inferir_flags_ergonomicos(
            lms,
            640,
            480,
            [{"logical": ("pantalla",), "bbox": (240, 110, 430, 250)}],
            (10.0, 90.0, 90.0, 0.0, 0.0),
        )

        self.assertTrue(flags["pantalla_baja"])
        self.assertFalse(flags["pantalla_dist_ok"])
        self.assertIn("cabeza agachada >30° en perfil", motivos)


if __name__ == "__main__":
    unittest.main()
