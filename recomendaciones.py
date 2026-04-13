def cfg_efectiva(cfg_base, flags_inferidos):
    cfg = dict(cfg_base or {})
    for key, value in (flags_inferidos or {}).items():
        if isinstance(value, bool) and key in cfg:
            cfg[key] = cfg[key] or value
    return cfg


def _codo_fuera_de_90(dato):
    ang_codo = dato.get("ang_codo")
    if ang_codo is None:
        return False
    try:
        ang_codo = float(ang_codo)
    except (TypeError, ValueError):
        return False
    return not (80.0 <= ang_codo <= 100.0)


def resumen_analisis(registro, cfg):
    partes = [
        (
            f"Silla={registro.get('total_silla', '')}, Pantalla/telefono={registro.get('tabla_B', '')}, "
            f"Mouse/teclado={registro.get('tabla_C', '')}, Combinado={registro.get('tabla_D', '')}."
        )
    ]

    if _codo_fuera_de_90(registro):
        partes.append(
            f"El codo está fuera del rango neutro de 80° a 100° ({float(registro.get('ang_codo')):.1f}°)."
        )
    if cfg.get("hombros_encogidos_silla"):
        partes.append("Se detectan hombros elevados o forzados por la relación silla/mesa.")
    elif registro.get("A3", 0) >= 2:
        partes.append("La corrección debe centrarse en apoyar mejor los antebrazos, no en relajar hombros si ya están neutros.")
    if abs(registro.get("ang_muneca", 0) or 0) > 15:
        partes.append("La muñeca se desvía al escribir y también aporta al riesgo.")

    return " ".join(partes)


def recomendaciones_alerta(dato, cfg):
    items = []

    def add_item(score, categoria, acciones, icono=None):
        if score <= 0:
            return
        acciones = [accion.strip() for accion in acciones if accion and accion.strip()]
        if not acciones:
            return
        items.append(
            {
                "score": int(score),
                "categoria": categoria,
                "acciones": acciones,
                "icono": icono,
            }
        )

    if dato.get("total_silla", 0) >= 4:
        acciones = []
        if not cfg.get("pie_llega_suelo", True):
            acciones.append("ajusta la altura de la silla hasta apoyar completamente los pies")
        if cfg.get("espacio_insuficiente_piernas"):
            acciones.append("libera espacio bajo la mesa para mantener rodillas cercanas a 90°")
        if not cfg.get("usa_respaldo", True):
            acciones.append("apoya la espalda en el respaldo")
        if not cfg.get("apoyo_lumbar_adecuado", True):
            acciones.append("corrige o agrega apoyo lumbar")
        if cfg.get("hombros_encogidos_silla") and _codo_fuera_de_90(dato):
            acciones.append("reajusta silla o mesa para relajar hombros y recuperar codos cerca de 90°")
        elif cfg.get("hombros_encogidos_silla"):
            acciones.append("baja o reajusta silla/mesa para relajar los hombros")
        elif _codo_fuera_de_90(dato):
            acciones.append("reajusta silla o reposabrazos hasta dejar los codos cerca de 90°")
        if not acciones:
            acciones.append("regula silla, asiento y respaldo hasta recuperar una postura neutra")
        add_item(dato.get("total_silla", 0), "Silla", acciones, "silla")

    if dato.get("B2", 0) >= 3:
        acciones = []
        if cfg.get("pantalla_elevada"):
            acciones.append("baja la pantalla hasta que el borde superior quede a la altura de los ojos")
        if cfg.get("pantalla_baja"):
            acciones.append("eleva la pantalla para evitar flexión del cuello")
        if not cfg.get("pantalla_dist_ok", True):
            acciones.append("coloca la pantalla entre 40 y 75 cm")
        if cfg.get("giro_otra_pantalla"):
            acciones.append("centra la pantalla principal frente al cuerpo")
        if cfg.get("pantalla_reflejos"):
            acciones.append("reduce reflejos cambiando el ángulo o la iluminación")
        if cfg.get("sin_portadocumentos"):
            acciones.append("usa portadocumentos a la misma altura de la pantalla")
        add_item(dato.get("B2", 0), "Pantalla", acciones, "pantalla")

    if dato.get("C2", 0) >= 3:
        acciones = []
        if abs(dato.get("ang_muneca", 0)) > 15:
            acciones.append("mantén la muñeca recta al escribir")
        if cfg.get("teclado_elevado_hombros"):
            acciones.append("baja el teclado para evitar hombros encogidos")
        if cfg.get("sin_soporte_teclado"):
            acciones.append("usa bandeja o soporte ajustable para teclado")
        if cfg.get("desviacion_escribir"):
            acciones.append("escribe con antebrazos alineados y sin desviación lateral")
        if cfg.get("alcance_sobre_cabeza"):
            acciones.append("reubica el teclado y objetos de uso frecuente por debajo del hombro")
        add_item(dato.get("C2", 0), "Teclado", acciones, "teclado")

    if dato.get("C1", 0) >= 3:
        acciones = []
        if not cfg.get("raton_alineado", True):
            acciones.append("acerca el mouse al cuerpo y alinéalo con el hombro")
        if cfg.get("raton_teclado_dif_altura"):
            acciones.append("coloca mouse y teclado a la misma altura")
        if cfg.get("agarre_pinza"):
            acciones.append("usa un mouse de tamaño adecuado para evitar agarre en pinza")
        if cfg.get("reposamanos_duro"):
            acciones.append("elimina puntos de presión en la base de la muñeca")
        add_item(dato.get("C1", 0), "Mouse", acciones, "mouse")

    if dato.get("B1", 0) >= 2:
        acciones = []
        if cfg.get("telefono_alejado"):
            acciones.append("acerca el teléfono a la zona de alcance")
        if cfg.get("sujecion_hombro_cuello"):
            acciones.append("deja de sujetar el teléfono con hombro o cuello")
        if cfg.get("sin_manos_libres"):
            acciones.append("usa manos libres o altavoz para llamadas prolongadas")
        add_item(dato.get("B1", 0), "Telefono", acciones, "telefono")

    if dato.get("A3", 0) >= 3:
        acciones = []
        if not cfg.get("tiene_reposabrazos", True):
            acciones.append("incorpora reposabrazos")
        if cfg.get("reposabrazos_altos_bajos") and _codo_fuera_de_90(dato):
            acciones.append(
                f"ajusta reposabrazos para acercar el codo a 90°; ahora está en {float(dato.get('ang_codo')):.1f}°"
            )
        elif cfg.get("reposabrazos_altos_bajos"):
            acciones.append("ajusta reposabrazos para mantener codos cerca de 90°")
        if cfg.get("reposabrazos_no_regulables") or not cfg.get("reposabrazos_ajustable", True):
            acciones.append("usa reposabrazos regulables")
        if cfg.get("bordes_afilados"):
            acciones.append("acolcha o cambia reposabrazos con bordes duros")
        if cfg.get("brazos_anchos"):
            acciones.append("reduce la separación de los reposabrazos")
        add_item(dato.get("A3", 0), "Reposabrazos", acciones, "reposabrazos")

    if dato.get("A1", 0) >= 3 or dato.get("A2", 0) >= 2 or dato.get("A4", 0) >= 3:
        acciones = []
        if dato.get("A1", 0) >= 3 or not cfg.get("pie_llega_suelo", True):
            acciones.append("corrige la altura del asiento para que pies y rodillas queden estables")
        if dato.get("A2", 0) >= 2:
            acciones.append("ajusta la profundidad del asiento dejando cerca de 8 cm detrás de la rodilla")
        if dato.get("A4", 0) >= 3:
            acciones.append("reclina y regula el respaldo para mantener apoyo lumbar y tronco estable")
        add_item(
            max(dato.get("A1", 0), dato.get("A2", 0), dato.get("A4", 0)),
            "Ajustes de asiento y respaldo",
            acciones,
            "ajustes de asiento y respaldo",
        )

    items.sort(key=lambda x: (-x["score"], x["categoria"]))
    if not items:
        return [
            {
                "score": 1,
                "categoria": "General",
                "acciones": ["Realiza una pausa activa y vuelve a una postura neutra."],
                "icono": "general",
            }
        ]
    return items[:5]
