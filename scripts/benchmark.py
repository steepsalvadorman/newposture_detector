"""
benchmark.py — Pruebas de rendimiento del núcleo de cálculo ROSA.

No requiere cámara ni GUI: mide la velocidad pura de los algoritmos.
Uso: python scripts/benchmark.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rosa.core.calculos import (
    calcular_ROSA_completo_v5,
    calcular_angulo,
    calcular_desviacion_vertical,
    puntuar_A1, puntuar_A2, puntuar_A3, puntuar_A4,
    puntuar_B1, puntuar_B2_pantalla,
    puntuar_C1_raton, puntuar_C2_teclado,
)

REPS_RAPIDO = 100_000
REPS_PESADO = 10_000

_KWARGS_ROSA = dict(
    ang_tronco=10.0, ang_rodilla=90.0, ang_codo=90.0,
    desv_cuello=5.0, ang_muneca=10.0,
    horas_silla=6.0, horas_telefono=1.0, horas_pantalla=6.0,
    horas_raton=6.0, horas_teclado=6.0,
    pie_llega_suelo=True, altura_regulable=True,
    espacio_insuficiente_piernas=False, dist_rodilla_cm=8.0,
    profundidad_regulable=True, tiene_reposabrazos=True,
    reposabrazos_ajustable=True, bordes_afilados=False,
    brazos_anchos=False, reposabrazos_no_regulables=False,
    reposabrazos_altos_bajos=False, usa_respaldo=True,
    apoyo_lumbar_adecuado=True, hombros_encogidos_silla=False,
    respaldo_no_regulable=False, telefono_alejado=False,
    sujecion_hombro_cuello=False, sin_manos_libres=False,
    pantalla_dist_ok=True, pantalla_baja=False, pantalla_elevada=False,
    dist_pantalla_mayor_75=False, giro_otra_pantalla=False,
    sin_portadocumentos=False, pantalla_reflejos=False,
    raton_alineado=True, agarre_pinza=False,
    raton_teclado_dif_altura=False, reposamanos_duro=False,
    desviacion_escribir=False, alcance_sobre_cabeza=False,
    teclado_elevado_hombros=False, sin_soporte_teclado=False,
)


def _medir(nombre, fn, reps):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    elapsed = time.perf_counter() - t0
    por_seg = reps / elapsed
    us = elapsed / reps * 1_000_000
    print(f"  {nombre:<38} {reps:>8,} iter  {por_seg:>12,.0f}/s  {us:>8.2f} µs")


def _sep(titulo=""):
    if titulo:
        print(f"\n{'─'*10} {titulo} {'─'*(47 - len(titulo))}")
    else:
        print("─" * 68)


def main():
    print("=" * 68)
    print("  Benchmark EvaluadorROSA — Núcleo de cálculo")
    print(f"  Python {sys.version.split()[0]}")
    print("=" * 68)

    _sep("Geometría")
    a = [0.0, 0.0]
    b = [1.0, 0.0]
    c = [1.0, 1.0]
    _medir("calcular_angulo", lambda: calcular_angulo(a, b, c), REPS_RAPIDO)
    _medir("calcular_desviacion_vertical", lambda: calcular_desviacion_vertical(a, c), REPS_RAPIDO)

    _sep("Puntuadores individuales")
    _medir("puntuar_A1",          lambda: puntuar_A1(90.0, True, True, False),  REPS_RAPIDO)
    _medir("puntuar_A2",          lambda: puntuar_A2(8.0, True),                REPS_RAPIDO)
    _medir("puntuar_A3",          lambda: puntuar_A3(True, True, False, False, False), REPS_RAPIDO)
    _medir("puntuar_A4",          lambda: puntuar_A4(10.0, True, True, False, False),  REPS_RAPIDO)
    _medir("puntuar_B1",          lambda: puntuar_B1(),                          REPS_RAPIDO)
    _medir("puntuar_B2_pantalla", lambda: puntuar_B2_pantalla(5.0, True, False, False, False, False, False, False), REPS_RAPIDO)
    _medir("puntuar_C1_raton",    lambda: puntuar_C1_raton(True, False, False, False), REPS_RAPIDO)
    _medir("puntuar_C2_teclado",  lambda: puntuar_C2_teclado(10.0, False, False, False, False), REPS_RAPIDO)

    _sep("ROSA completo (todas las tablas)")
    _medir("calcular_ROSA_completo_v5", lambda: calcular_ROSA_completo_v5(**_KWARGS_ROSA), REPS_PESADO)

    _sep()
    print(f"  Intervalo de evaluación configurado: 6 s")
    from rosa.core.calculos import INTERVALO_EVAL_SEG
    t0 = time.perf_counter()
    calcular_ROSA_completo_v5(**_KWARGS_ROSA)
    single_us = (time.perf_counter() - t0) * 1_000_000
    overhead_pct = single_us / (INTERVALO_EVAL_SEG * 1_000_000) * 100
    print(f"  Un cálculo tarda ~{single_us:.1f} µs = {overhead_pct:.6f}% del intervalo")
    print("=" * 68)


if __name__ == "__main__":
    main()
