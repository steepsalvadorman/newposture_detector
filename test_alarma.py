import tkinter as tk

from gui_principal import mostrar_overlay_alarma, recomendaciones_alerta


CASOS = {
    "Prioridad teclado": (
        {
            "rosa": 6,
            "total_silla": 4,
            "B1": 1,
            "B2": 3,
            "C1": 3,
            "C2": 5,
            "A1": 2,
            "A2": 1,
            "A3": 2,
            "A4": 3,
            "ang_muneca": 24.0,
        },
        {
            "pie_llega_suelo": True,
            "espacio_insuficiente_piernas": False,
            "usa_respaldo": True,
            "apoyo_lumbar_adecuado": True,
            "hombros_encogidos_silla": False,
            "pantalla_elevada": False,
            "pantalla_baja": True,
            "pantalla_dist_ok": False,
            "giro_otra_pantalla": True,
            "pantalla_reflejos": False,
            "sin_portadocumentos": False,
            "teclado_elevado_hombros": True,
            "sin_soporte_teclado": True,
            "desviacion_escribir": True,
            "alcance_sobre_cabeza": False,
            "raton_alineado": False,
            "raton_teclado_dif_altura": True,
            "agarre_pinza": False,
            "reposamanos_duro": False,
            "telefono_alejado": False,
            "sujecion_hombro_cuello": False,
            "sin_manos_libres": False,
            "tiene_reposabrazos": True,
            "reposabrazos_altos_bajos": False,
            "reposabrazos_no_regulables": False,
            "reposabrazos_ajustable": True,
            "bordes_afilados": False,
            "brazos_anchos": False,
        },
    ),
    "Prioridad silla": (
        {
            "rosa": 7,
            "total_silla": 6,
            "B1": 2,
            "B2": 2,
            "C1": 2,
            "C2": 2,
            "A1": 4,
            "A2": 2,
            "A3": 3,
            "A4": 4,
            "ang_muneca": 8.0,
        },
        {
            "pie_llega_suelo": False,
            "espacio_insuficiente_piernas": True,
            "usa_respaldo": False,
            "apoyo_lumbar_adecuado": False,
            "hombros_encogidos_silla": True,
            "pantalla_elevada": False,
            "pantalla_baja": False,
            "pantalla_dist_ok": True,
            "giro_otra_pantalla": False,
            "pantalla_reflejos": False,
            "sin_portadocumentos": False,
            "teclado_elevado_hombros": False,
            "sin_soporte_teclado": False,
            "desviacion_escribir": False,
            "alcance_sobre_cabeza": False,
            "raton_alineado": True,
            "raton_teclado_dif_altura": False,
            "agarre_pinza": False,
            "reposamanos_duro": False,
            "telefono_alejado": True,
            "sujecion_hombro_cuello": False,
            "sin_manos_libres": True,
            "tiene_reposabrazos": True,
            "reposabrazos_altos_bajos": True,
            "reposabrazos_no_regulables": True,
            "reposabrazos_ajustable": False,
            "bordes_afilados": True,
            "brazos_anchos": False,
        },
    ),
}


def _formatear_recomendaciones(recomendaciones):
    lineas = []
    for item in recomendaciones:
        lineas.append(f"{item['categoria']} (puntaje {item['score']}):")
        for accion in item.get("acciones", []):
            lineas.append(f"- {accion}")
    return "\n".join(lineas)


class PruebaAlarmaApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Prueba de alarma ROSA")
        self.geometry("760x360")
        self.configure(bg="#111318")
        self.caso_actual = tk.StringVar(value="Prioridad teclado")

        contenedor = tk.Frame(self, bg="#111318", padx=24, pady=24)
        contenedor.pack(fill="both", expand=True)

        tk.Label(
            contenedor,
            text="Prueba visual de la alarma ergonomica",
            font=("Consolas", 16, "bold"),
            bg="#111318",
            fg="#F0F4FF",
        ).pack(pady=(0, 10))

        tk.Label(
            contenedor,
            text="Selecciona un escenario y abre la alerta para ver el orden real de recomendaciones.",
            font=("Consolas", 10),
            bg="#111318",
            fg="#9BAEC8",
            justify="center",
        ).pack(pady=(0, 16))

        selector = tk.Frame(contenedor, bg="#111318")
        selector.pack(pady=(0, 16))
        for nombre in CASOS:
            tk.Radiobutton(
                selector,
                text=nombre,
                value=nombre,
                variable=self.caso_actual,
                command=self.actualizar_preview,
                font=("Consolas", 10),
                bg="#111318",
                fg="#F0F4FF",
                selectcolor="#181C22",
                activebackground="#111318",
                activeforeground="#00C8FF",
            ).pack(anchor="w")

        tk.Button(
            contenedor,
            text="Mostrar alarma",
            command=self.mostrar_alarma,
            font=("Consolas", 12, "bold"),
            bg="#FF4757",
            fg="white",
            activebackground="#C73A47",
            activeforeground="white",
            relief="flat",
            padx=18,
            pady=10,
        ).pack()

        self.lbl_preview = tk.Label(
            contenedor,
            text="",
            font=("Consolas", 9),
            bg="#111318",
            fg="#9BAEC8",
            justify="left",
            anchor="w",
            wraplength=680,
        )
        self.lbl_preview.pack(fill="x", pady=(18, 0))

        tk.Label(
            contenedor,
            text="La alerta se cierra moviendo el mouse, haciendo clic o con Escape.",
            font=("Consolas", 9),
            bg="#111318",
            fg="#56657A",
        ).pack(pady=(18, 0))

        self.actualizar_preview()

    def _recomendaciones_caso(self):
        dato, cfg = CASOS[self.caso_actual.get()]
        recomendaciones = recomendaciones_alerta(dato, cfg)
        print(f"\n[{self.caso_actual.get()}]")
        print(_formatear_recomendaciones(recomendaciones))
        return recomendaciones

    def actualizar_preview(self):
        recomendaciones = self._recomendaciones_caso()
        self.lbl_preview.config(text="Vista previa del orden que se mostrara en la alerta:\n" + _formatear_recomendaciones(recomendaciones))

    def mostrar_alarma(self):
        mostrar_overlay_alarma(self, self._recomendaciones_caso())


if __name__ == "__main__":
    app = PruebaAlarmaApp()
    app.mainloop()
