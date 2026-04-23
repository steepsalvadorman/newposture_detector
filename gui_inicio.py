import os
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2

from excel_modelo import descargar_modelo
from tablas_calculos import C, MAX_CAMARAS_SCAN


def detectar_camaras():
    camaras = []
    for i in range(MAX_CAMARAS_SCAN):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        try:
            prop = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
            if prop is not None:
                cap.set(prop, 700)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    camaras.append((i, f"Camara {i}"))
        finally:
            try:
                cap.release()
            except Exception:
                pass
    return camaras if camaras else [(0, "Default")]


class DialogoInicio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ROSA Ergonomy Detection")
        self.geometry("540x720")
        self.minsize(540, 660)
        self.configure(bg=C["bg_deep"])
        self.resultado = None
        self._vars = {}
        self._build()

    def _bv(self, nombre, default=False):
        var = tk.BooleanVar(value=default)
        self._vars[nombre] = var
        return var

    def _dv(self, nombre, default=8.0):
        var = tk.DoubleVar(value=default)
        self._vars[nombre] = var
        return var

    def _build(self):
        header = tk.Frame(self, bg=C["bg_deep"])
        header.pack(fill="x", padx=18, pady=(16, 8))
        tk.Label(
            header,
            text="Sistema automatizado de ergonomía con cámara web",
            font=("Consolas", 16, "bold"),
            bg=C["bg_deep"],
            fg=C["accent"],
        ).pack(anchor="w")
        self.lbl_modelo = tk.Label(
            self,
            text="Verificando modelo...",
            font=("Consolas", 8),
            bg=C["bg_deep"],
            fg=C["warn"],
        )
        self.lbl_modelo.pack(anchor="w", padx=18, pady=(0, 8))

        outer = tk.Frame(self, bg=C["bg_deep"])
        outer.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        canvas = tk.Canvas(outer, bg=C["bg_deep"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        body = tk.Frame(canvas, bg=C["bg_card"])
        canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind_all(
            "<MouseWheel>",
            lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
        )

        self._seccion(body, "General")
        self._campo_texto(body, "Nombre del proyecto", "Evaluacion ergonomica ROSA", "proyecto")
        self._campo_texto(body, "Evaluador", "Jaime Cesar Tarazona Tinoco", "evaluador")
        self._campo_texto(body, "Empresa", "CHYC Ingenieros SAC", "empresa")
        self._campo_texto(body, "Nombre del trabajador", "Trabajador 01", "trabajador")
        self._campo_horas(body)
        self._campo_camara(body)

        self._seccion(body, "Consideraciones de la evaluacion ergonomica")
        self._seccion(body, "Silla")
        self._check(body, "El pie llega al suelo", self._bv("pie_llega_suelo", True))
        self._check(body, "Altura de la silla regulable", self._bv("altura_regulable", True))
        self._check(body, "Espacio insuficiente para las piernas", self._bv("espacio_insuficiente_piernas"))
        self._campo_distancia(body)
        self._check(
            body,
            "Profundidad del asiento regulable",
            self._bv("profundidad_regulable", True),
        )
        self._check(body, "Tiene reposabrazos", self._bv("tiene_reposabrazos", True))
        self._check(
            body,
            "Reposabrazos ajustable al codo",
            self._bv("reposabrazos_ajustable", True),
        )
        self._check(
            body,
            "Reposabrazos altos o bajos (codos sin apoyar / hombros encogidos)",
            self._bv("reposabrazos_altos_bajos"),
        )
        self._check(body, "Reposabrazos con bordes duros", self._bv("bordes_afilados"))
        self._check(body, "Reposabrazos demasiado anchos", self._bv("brazos_anchos"))
        self._check(
            body,
            "Reposabrazos no regulables",
            self._bv("reposabrazos_no_regulables"),
        )
        self._check(body, "Usa el respaldo", self._bv("usa_respaldo", True))
        self._check(
            body,
            "Apoyo lumbar adecuado",
            self._bv("apoyo_lumbar_adecuado", True),
        )
        self._check(
            body,
            "Hombros encogidos por mesa o silla",
            self._bv("hombros_encogidos_silla"),
        )
        self._check(
            body,
            "Respaldo no regulable",
            self._bv("respaldo_no_regulable"),
        )

        self._seccion(body, "Pantalla")
        self._check(
            body,
            "Pantalla entre 40 y 75 cm y a la altura de los ojos",
            self._bv("pantalla_dist_ok", True),
        )
        self._check(
            body,
            "Pantalla baja, por debajo de 30° desde la linea de vision",
            self._bv("pantalla_baja"),
        )
        self._check(
            body,
            "Pantalla elevada y obliga a extender el cuello",
            self._bv("pantalla_elevada"),
        )
        self._check(body, "Pantalla a mas de 75 cm", self._bv("dist_pantalla_mayor_75"))
        self._check(body, "Gira el cuello para ver otra pantalla", self._bv("giro_otra_pantalla"))
        self._check(body, "No usa portadocumentos", self._bv("sin_portadocumentos"))
        self._check(body, "Pantalla con reflejos", self._bv("pantalla_reflejos"))

        self._seccion(body, "Mouse")
        self._check(
            body,
            "Mouse alineado con el hombro y dentro del alcance",
            self._bv("raton_alineado", True),
        )
        self._check(body, "Agarre en pinza por mouse pequeño", self._bv("agarre_pinza"))
        self._check(
            body,
            "Mouse y teclado a distinta altura",
            self._bv("raton_teclado_dif_altura"),
        )
        self._check(
            body,
            "Reposamanos duro o con presión",
            self._bv("reposamanos_duro"),
        )

        self._seccion(body, "Teclado")
        self._check(
            body,
            "Desviación al escribir",
            self._bv("desviacion_escribir"),
        )
        self._check(
            body,
            "Alcanza objetos por encima de la cabeza",
            self._bv("alcance_sobre_cabeza"),
        )
        self._check(
            body,
            "Teclado elevado con hombros encogidos",
            self._bv("teclado_elevado_hombros"),
        )
        self._check(
            body,
            "Teclado sin soporte ajustable",
            self._bv("sin_soporte_teclado"),
        )

        footer = tk.Frame(self, bg=C["bg_deep"])
        footer.pack(fill="x", padx=18, pady=(0, 16))

        self.btn_iniciar = tk.Button(
            footer,
            text="INICIAR EVALUACION",
            command=self._ok,
            state="disabled",
            font=("Consolas", 11, "bold"),
            bg=C["accent"],
            fg="white",
            relief="flat",
            cursor="hand2",
        )
        self.btn_iniciar.pack(fill="x", ipady=8)

        threading.Thread(target=self._verificar_modelo, daemon=True).start()

    def _seccion(self, parent, titulo):
        tk.Label(
            parent,
            text=titulo,
            font=("Consolas", 10, "bold"),
            bg=C["bg_card"],
            fg=C["accent"],
            anchor="w",
        ).pack(fill="x", padx=12, pady=(12, 6))

    def _campo_texto(self, parent, label, default, key):
        tk.Label(
            parent,
            text=label,
            font=("Consolas", 9),
            bg=C["bg_card"],
            fg=C["text_mid"],
        ).pack(anchor="w", padx=12)
        entry = tk.Entry(
            parent,
            font=("Consolas", 10),
            bg=C["bg_border"],
            fg=C["text_hi"],
            insertbackground="white",
            relief="flat",
        )
        entry.insert(0, default)
        entry.pack(fill="x", padx=12, pady=(2, 6))
        self._vars[key] = entry
        if key == "trabajador":
            self.ent_nombre = entry

    def _campo_horas(self, parent):
        tk.Label(
            parent,
            text="Horas diarias frente al equipo",
            font=("Consolas", 9),
            bg=C["bg_card"],
            fg=C["text_mid"],
        ).pack(anchor="w", padx=12)
        self.spin_horas = ttk.Spinbox(
            parent,
            from_=0.5,
            to=12,
            increment=0.5,
            width=10,
            font=("Consolas", 10),
        )
        self.spin_horas.set("6")
        self.spin_horas.pack(anchor="w", padx=12, pady=(2, 6))
        aviso = tk.Frame(
            parent,
            bg=C["bg_panel"],
            highlightthickness=1,
            highlightbackground=C["accent2"],
        )
        aviso.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(
            aviso,
            text="Consideracion tabla F",
            font=("Consolas", 9, "bold"),
            bg=C["bg_panel"],
            fg=C["accent"],
        ).pack(anchor="w", padx=10, pady=(8, 2))
        tk.Label(
            aviso,
            text=">4 h = +1    |    1-4 h = 0    |    <1 h = -1",
            font=("Consolas", 10, "bold"),
            bg=C["bg_panel"],
            fg=C["text_hi"],
        ).pack(anchor="w", padx=10, pady=(0, 8))

    def _campo_camara(self, parent):
        tk.Label(
            parent,
            text="Camara",
            font=("Consolas", 9),
            bg=C["bg_card"],
            fg=C["text_mid"],
        ).pack(anchor="w", padx=12)
        self._camaras = detectar_camaras()
        self.combo_cam = ttk.Combobox(
            parent,
            values=[f"{idx}: {nombre}" for idx, nombre in self._camaras],
            state="readonly",
            font=("Consolas", 10),
        )
        self.combo_cam.current(0)
        self.combo_cam.pack(fill="x", padx=12, pady=(2, 6))

    def _campo_horas_factor(self, parent, label, key, default):
        tk.Label(
            parent,
            text=label,
            font=("Consolas", 9),
            bg=C["bg_card"],
            fg=C["text_mid"],
        ).pack(anchor="w", padx=12)
        self._dv(key, default)
        spin = ttk.Spinbox(
            parent,
            from_=0,
            to=12,
            increment=0.5,
            width=10,
            font=("Consolas", 10),
            textvariable=self._vars[key],
        )
        spin.pack(anchor="w", padx=12, pady=(2, 6))

    def _campo_distancia(self, parent):
        tk.Label(
            parent,
            text="Distancia asiento-rodilla (cm)",
            font=("Consolas", 9),
            bg=C["bg_card"],
            fg=C["text_mid"],
        ).pack(anchor="w", padx=12)
        self._dv("dist_rodilla_cm", 8.0)
        self.spin_dist = ttk.Spinbox(
            parent,
            from_=0,
            to=30,
            increment=0.5,
            width=10,
            font=("Consolas", 10),
            textvariable=self._vars["dist_rodilla_cm"],
        )
        self.spin_dist.pack(anchor="w", padx=12, pady=(2, 6))
        tk.Label(
            parent,
            text="Postura neutra: exactamente 8 cm",
            font=("Consolas", 8),
            bg=C["bg_card"],
            fg=C["text_lo"],
        ).pack(anchor="w", padx=12)

    def _check(self, parent, texto, var):
        tk.Checkbutton(
            parent,
            text=texto,
            variable=var,
            font=("Consolas", 9),
            bg=C["bg_card"],
            fg=C["text_hi"],
            selectcolor=C["bg_border"],
            activebackground=C["bg_card"],
            activeforeground=C["accent"],
            anchor="w",
            wraplength=470,
        ).pack(fill="x", padx=12, pady=1)

    def _verificar_modelo(self):
        def status(msg):
            try:
                self.lbl_modelo.config(text=msg)
            except Exception:
                pass

        ok = descargar_modelo(status)
        try:
            if ok:
                self.lbl_modelo.config(text="Modelo listo", fg=C["ok"])
                self.btn_iniciar.config(state="normal")
            else:
                self.lbl_modelo.config(
                    text="No se pudo descargar el modelo",
                    fg=C["danger"],
                )
        except Exception:
            pass

    def _ok(self):
        proyecto = self._vars["proyecto"].get().strip() or "Evaluacion ergonomica ROSA"
        evaluador = self._vars["evaluador"].get().strip() or "Jaime Cesar Tarazona Tinoco"
        empresa = self._vars["empresa"].get().strip() or "CHYC Ingenieros SAC"
        nombre = self.ent_nombre.get().strip() or "Trabajador"
        horas = float(self.spin_horas.get())
        idx_cam = self._camaras[self.combo_cam.current()][0]
        ruta_xlsx = os.path.join(
            os.path.expanduser("~"),
            "Desktop",
            f"ROSA_{nombre.replace(' ', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
        )

        cfg = {}
        for k, v in self._vars.items():
            if isinstance(v, tk.BooleanVar):
                cfg[k] = v.get()
            elif isinstance(v, tk.DoubleVar):
                cfg[k] = float(v.get())
        cfg["horas_silla"] = horas
        cfg["horas_pantalla"] = horas
        cfg["horas_raton"] = horas
        cfg["horas_teclado"] = horas
        cfg["nombre_proyecto"] = proyecto
        cfg["evaluador"] = evaluador
        cfg["empresa"] = empresa
        cfg["trabajador"] = nombre

        self.resultado = (nombre, horas, idx_cam, ruta_xlsx, cfg)
        self.destroy()
