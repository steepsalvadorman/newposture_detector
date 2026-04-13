import unittest

from tablas_calculos import (
    calcular_ROSA_completo_v5,
    puntuar_A1,
    puntuar_A2,
    puntuar_A4,
    puntuar_B2_pantalla,
    puntuar_C1_raton,
    puntuar_C2_teclado,
)


class TestTablasCalculos(unittest.TestCase):
    def test_a1_solo_90_es_neutro(self):
        self.assertEqual(puntuar_A1(90.0, True, True), 1)
        self.assertEqual(puntuar_A1(89.4, True, True), 2)
        self.assertEqual(puntuar_A1(91.0, True, True), 2)
        self.assertEqual(puntuar_A1(90.0, False, True), 3)

    def test_a2_solo_8_cm_es_neutro(self):
        self.assertEqual(puntuar_A2(8.0, True), 1)
        self.assertEqual(puntuar_A2(7.0, True), 2)
        self.assertEqual(puntuar_A2(9.0, True), 2)
        self.assertEqual(puntuar_A2(8.0, False), 2)

    def test_a3_criterios_adicionales_solo_desde_ui(self):
        from tablas_calculos import puntuar_A3

        self.assertEqual(
            puntuar_A3(
                tiene_reposabrazos=True,
                reposabrazos_ajustable=False,
                bordes_afilados=False,
                demasiado_anchos=False,
                no_regulables=False,
                reposabrazos_altos_bajos=False,
            ),
            1,
        )
        self.assertEqual(
            puntuar_A3(
                tiene_reposabrazos=True,
                reposabrazos_ajustable=True,
                bordes_afilados=False,
                demasiado_anchos=False,
                no_regulables=True,
                reposabrazos_altos_bajos=False,
            ),
            2,
        )

    def test_a4_rangos_y_criterios_adicionales(self):
        self.assertEqual(puntuar_A4(6.0, True, True, False, False), 1)
        self.assertEqual(puntuar_A4(5.0, True, True, False, False), 2)
        self.assertEqual(puntuar_A4(20.0, True, True, False, False), 2)
        self.assertEqual(puntuar_A4(12.0, True, False, False, False), 2)
        self.assertEqual(puntuar_A4(12.0, False, True, False, False), 2)
        self.assertEqual(puntuar_A4(12.0, True, True, True, True), 3)

    def test_b2_c1_c2_permiten_cero_por_factor_tiempo(self):
        self.assertEqual(
            puntuar_B2_pantalla(
                0.0,
                pantalla_dist_ok=True,
                pantalla_baja=False,
                pantalla_elevada=False,
                dist_mayor_75=False,
                giro_cuello_otra_pantalla=False,
                sin_portadocumentos=False,
                con_reflejos=False,
                factor_t=-1,
            ),
            0,
        )
        self.assertEqual(
            puntuar_C1_raton(
                raton_alineado_hombro=True,
                agarre_pinza=False,
                raton_teclado_diferente_altura=False,
                reposamanos_duro=False,
                factor_t=-1,
            ),
            0,
        )
        self.assertEqual(
            puntuar_C2_teclado(
                ang_muneca=0.0,
                desviacion_al_escribir=False,
                alcance_sobre_cabeza=False,
                teclado_elevado_hombros=False,
                sin_soporte_ajustable=False,
                factor_t=-1,
            ),
            0,
        )

    def test_b2_no_suma_por_distancia_mayor_75(self):
        self.assertEqual(
            puntuar_B2_pantalla(
                0.0,
                pantalla_dist_ok=False,
                pantalla_baja=False,
                pantalla_elevada=False,
                dist_mayor_75=True,
                giro_cuello_otra_pantalla=False,
                sin_portadocumentos=False,
                con_reflejos=False,
                factor_t=0,
            ),
            2,
        )

    def test_ejemplo_oficial_ntp_1173(self):
        resultado = calcular_ROSA_completo_v5(
            ang_tronco=12.0,
            ang_rodilla=100.0,
            ang_codo=90.0,
            desv_cuello=0.0,
            ang_muneca=0.0,
            horas_silla=6.0,
            horas_telefono=2.0,
            horas_pantalla=6.0,
            horas_raton=6.0,
            horas_teclado=2.0,
            pie_llega_suelo=True,
            altura_regulable=False,
            espacio_insuficiente_piernas=False,
            dist_rodilla_cm=7.5,
            profundidad_regulable=True,
            tiene_reposabrazos=True,
            reposabrazos_ajustable=True,
            bordes_afilados=False,
            brazos_anchos=False,
            reposabrazos_no_regulables=True,
            reposabrazos_altos_bajos=True,
            usa_respaldo=True,
            apoyo_lumbar_adecuado=False,
            hombros_encogidos_silla=False,
            respaldo_no_regulable=True,
            telefono_alejado=False,
            sujecion_hombro_cuello=False,
            sin_manos_libres=False,
            pantalla_dist_ok=True,
            pantalla_baja=False,
            pantalla_elevada=False,
            dist_pantalla_mayor_75=False,
            giro_otra_pantalla=False,
            sin_portadocumentos=True,
            pantalla_reflejos=False,
            raton_alineado=False,
            agarre_pinza=False,
            raton_teclado_dif_altura=False,
            reposamanos_duro=True,
            desviacion_escribir=False,
            alcance_sobre_cabeza=False,
            teclado_elevado_hombros=False,
            sin_soporte_teclado=False,
        )

        self.assertEqual(resultado["A1"], 3)
        self.assertEqual(resultado["A2"], 1)
        self.assertEqual(resultado["A3"], 3)
        self.assertEqual(resultado["A4"], 3)
        self.assertEqual(resultado["tabla_A"], 5)
        self.assertEqual(resultado["total_silla"], 6)
        self.assertEqual(resultado["B1"], 1)
        self.assertEqual(resultado["B2"], 3)
        self.assertEqual(resultado["tabla_B"], 2)
        self.assertEqual(resultado["C1"], 4)
        self.assertEqual(resultado["C2"], 1)
        self.assertEqual(resultado["tabla_C"], 4)
        self.assertEqual(resultado["tabla_D"], 4)
        self.assertEqual(resultado["rosa"], 6)


if __name__ == "__main__":
    unittest.main()
