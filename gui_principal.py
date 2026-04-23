import copy
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageDraw, ImageTk

from camara_detection import HiloCamara
from excel_modelo import agregar_registro_excel, crear_excel_nuevo
from recomendaciones import cfg_efectiva, recomendaciones_alerta
from tablas_calculos import C, GUARDADO_SEG, INTERVALO_EVAL_SEG, NIVEL_ACCION, NIVELES, WINSOUND_OK

if WINSOUND_OK:
    import winsound


RETARDO_ALARMA_SEG = 10


def nivel_accion(score):
    key = min(max(score, 1), 5)
    return NIVELES[key]


def motivo_a1(dato, cfg):
    motivos = []
    if cfg["espacio_insuficiente_piernas"]:
        motivos.append("espacio insuficiente para las piernas")
    if not cfg["pie_llega_suelo"]:
        motivos.append("pies sin apoyo")
    if not cfg["altura_regulable"]:
        motivos.append("altura no regulable")
    if dato["A1"] >= 2 and not motivos:
        motivos.append("rodillas fuera de 90°")
    if motivos:
        return ", ".join(motivos)
    if dato["A1"] == 2:
        return "rodillas fuera de 90°"
    return "rodillas en rango neutro"


def motivo_a2(dato, cfg):
    return "distancia asiento-rodilla correcta" if dato["A2"] == 1 else f"distancia {cfg['dist_rodilla_cm']:.1f} cm fuera de 8 cm"


def motivo_a3(dato, cfg):
    motivos = []
    if not cfg["tiene_reposabrazos"]:
        motivos.append("sin reposabrazos")
    if cfg["reposabrazos_altos_bajos"]:
        motivos.append("reposabrazos altos o bajos")
    if not cfg["reposabrazos_ajustable"] or cfg["reposabrazos_no_regulables"]:
        motivos.append("no ajustables")
    if cfg["bordes_afilados"]:
        motivos.append("bordes duros")
    if cfg["brazos_anchos"]:
        motivos.append("demasiado anchos")
    return ", ".join(motivos) if motivos else "reposabrazos en condición neutra"


def motivo_a4(dato, cfg):
    motivos = []
    if not cfg["usa_respaldo"]:
        motivos.append("sin usar respaldo")
    if not cfg["apoyo_lumbar_adecuado"]:
        motivos.append("sin apoyo lumbar")
    if dato["ang_tronco"] <= 5 or dato["ang_tronco"] >= 20:
        motivos.append("inclinación fuera de 95°-110°")
    if cfg["hombros_encogidos_silla"]:
        motivos.append("hombros encogidos")
    if cfg["respaldo_no_regulable"]:
        motivos.append("respaldo no regulable")
    return ", ".join(motivos) if motivos else "respaldo y tronco en rango neutro"


def motivo_b1(dato, cfg):
    motivos = []
    if cfg.get("telefono_alejado"):
        motivos.append("teléfono alejado >30 cm")
    if cfg.get("sujecion_hombro_cuello"):
        motivos.append("sujeción con hombro/cuello")
    if cfg.get("sin_manos_libres"):
        motivos.append("sin manos libres")
    return ", ".join(motivos) if motivos else "teléfono en postura neutra"


def motivo_b2(dato, cfg):
    motivos = []
    if cfg["pantalla_elevada"]:
        motivos.append("pantalla alta con extensión de cuello")
    elif cfg["pantalla_baja"]:
        motivos.append("pantalla baja")
    elif not cfg["pantalla_dist_ok"]:
        motivos.append("pantalla fuera de postura neutra")
    if cfg["giro_otra_pantalla"]:
        motivos.append("giro de cuello")
    if cfg["sin_portadocumentos"]:
        motivos.append("sin portadocumentos")
    if cfg["pantalla_reflejos"]:
        motivos.append("reflejos")
    return ", ".join(motivos) if motivos else "pantalla en rango neutro"


def motivo_c1(dato, cfg):
    motivos = []
    if not cfg["raton_alineado"]:
        motivos.append("ratón fuera de alcance")
    if cfg["agarre_pinza"]:
        motivos.append("agarre en pinza")
    if cfg["raton_teclado_dif_altura"]:
        motivos.append("ratón y teclado a distinta altura")
    if cfg["reposamanos_duro"]:
        motivos.append("punto de presión")
    return ", ".join(motivos) if motivos else "ratón en rango neutro"


def motivo_c2(dato, cfg):
    motivos = []
    if abs(dato["ang_muneca"]) > 15:
        motivos.append("muñeca desviada")
    if cfg["desviacion_escribir"]:
        motivos.append("desviación al escribir")
    if cfg["teclado_elevado_hombros"]:
        motivos.append("teclado elevado con hombros encogidos")
    if cfg["alcance_sobre_cabeza"]:
        motivos.append("alcance sobre cabeza")
    if cfg["sin_soporte_teclado"]:
        motivos.append("sin soporte ajustable")
    return ", ".join(motivos) if motivos else "teclado en rango neutro"


def _crear_icono_categoria(categoria, size=60):
    bg = {"silla": "#243447", "mouse": "#173A2A", "teclado": "#3A2417"}.get(categoria, "#2B2B2B")
    fg = "#F7F4EA"
    ac = "#FFCF33"
    img = Image.new("RGBA", (size, size), bg)
    draw = ImageDraw.Draw(img)
    if categoria == "silla":
        draw.rounded_rectangle((18, 12, 40, 24), radius=4, outline=fg, width=3)
        draw.line((20, 24, 20, 40), fill=fg, width=3)
        draw.line((38, 24, 48, 36), fill=fg, width=3)
        draw.line((20, 40, 12, 50), fill=fg, width=3)
        draw.line((20, 40, 30, 50), fill=fg, width=3)
        draw.line((48, 36, 40, 48), fill=fg, width=3)
        draw.line((48, 36, 54, 42), fill=fg, width=3)
        draw.arc((10, 6, 50, 46), start=210, end=320, fill=ac, width=3)
    elif categoria == "mouse":
        draw.rounded_rectangle((18, 10, 42, 48), radius=12, outline=fg, width=3)
        draw.line((30, 14, 30, 26), fill=fg, width=3)
        draw.line((18, 30, 42, 30), fill=fg, width=2)
        draw.arc((8, 8, 52, 52), start=35, end=140, fill=ac, width=3)
    elif categoria == "teclado":
        draw.rounded_rectangle((10, 18, 50, 42), radius=4, outline=fg, width=3)
        for y in (23, 29, 35):
            for x in (15, 21, 27, 33, 39, 45):
                draw.rectangle((x, y, x + 3, y + 3), fill=fg)
        draw.line((14, 46, 46, 46), fill=ac, width=3)
    else:
        draw.ellipse((14, 14, 46, 46), outline=fg, width=3)
    return ImageTk.PhotoImage(img)


def _ruta_recurso(*partes):
    base_dir = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, *partes)


def _cargar_icono_categoria(categoria, size=60):
    categoria = (categoria or "").strip().lower()
    rutas = {
        "silla": _ruta_recurso("assets", "alert_icons", "silla.png"),
        "mouse": _ruta_recurso("assets", "alert_icons", "mouse.png"),
        "teclado": _ruta_recurso("assets", "alert_icons", "teclado.png"),
        "pantalla": _ruta_recurso("assets", "alert_icons", "pantalla.png"),
        "telefono": _ruta_recurso("assets", "alert_icons", "telefono.png"),
        "reposabrazos": _ruta_recurso("assets", "alert_icons", "reposabrazos.png"),
        "ajustes de asiento y respaldo": _ruta_recurso("assets", "alert_icons", "ajustes_asiento_respaldo.png"),
        "general": _ruta_recurso("assets", "alert_icons", "general.png"),
    }
    ruta = rutas.get(categoria)
    if ruta and os.path.exists(ruta):
        try:
            img = Image.open(ruta).convert("RGBA")
            img.thumbnail((size, size), Image.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception:
            pass
    return _crear_icono_categoria(categoria, size=size)


def mostrar_overlay_alarma(parent, recomendaciones, on_close=None):
    overlay = tk.Toplevel(parent)
    overlay.attributes("-fullscreen", True)
    overlay.attributes("-topmost", True)
    overlay.configure(bg="#8B0000")
    overlay.overrideredirect(True)
    overlay.grab_set()
    overlay.focus_force()
    overlay._iconos_alerta = []

    def _limitar(valor, minimo, maximo):
        return max(minimo, min(valor, maximo))

    screen_w = overlay.winfo_screenwidth()
    screen_h = overlay.winfo_screenheight()
    es_laptop_baja = screen_h <= 800
    es_ultrawide = screen_w >= 2200

    panel_w = _limitar(int(screen_w * (0.8 if es_laptop_baja else 0.72)), 760, 1380 if es_ultrawide else 1480)
    panel_h = _limitar(int(screen_h * (0.82 if es_laptop_baja else 0.76)), 500, 900)
    pad_x = _limitar(int(panel_w * (0.03 if es_laptop_baja else 0.035)), 18, 42)
    title_font = _limitar(int(panel_w * 0.026), 20, 34)
    subtitle_font = _limitar(int(panel_w * 0.014), 12, 18)
    section_font = _limitar(int(panel_w * 0.0115), 10, 15)
    action_font = _limitar(int(panel_w * 0.0098), 9, 13)
    footer_font = _limitar(int(panel_w * 0.011), 10, 14)
    icon_size = _limitar(int(panel_w * 0.04), 40, 68)
    subtitle_wrap = _limitar(int(panel_w * (0.88 if es_ultrawide else 0.9)), 360, 980)
    action_wrap = _limitar(int(panel_w * (0.64 if es_ultrawide else 0.72)), 300, 920)
    top_pad = _limitar(int(panel_h * (0.045 if es_laptop_baja else 0.06)), 20, 42)
    bottom_pad = _limitar(int(panel_h * 0.03), 12, 24)
    row_pad = 8 if es_laptop_baja else 12
    row_gap = 4 if es_laptop_baja else 6

    def _reproducir_sonido_alerta():
        if WINSOUND_OK:
            try:
                for frecuencia, duracion in ((1200, 180), (900, 220), (1200, 180)):
                    winsound.Beep(frecuencia, duracion)
                    time.sleep(0.05)
                return
            except Exception:
                try:
                    winsound.MessageBeep(winsound.MB_ICONHAND)
                    return
                except Exception:
                    pass
        try:
            overlay.bell()
            time.sleep(0.15)
            overlay.bell()
        except Exception:
            pass

    panel = tk.Frame(overlay, bg="#120000", highlightthickness=2, highlightbackground="#FFCF33")
    panel.place(relx=0.5, rely=0.5, anchor="center", width=panel_w, height=panel_h)
    panel.grid_columnconfigure(0, weight=1)
    panel.grid_rowconfigure(2, weight=1)

    lbl_titulo = tk.Label(
        panel,
        text="RIESGO ERGONOMICO ALTO",
        font=("Consolas", title_font, "bold"),
        bg="#120000",
        fg="#FFCF33",
    )
    lbl_titulo.grid(row=0, column=0, padx=pad_x, pady=(top_pad, 14 if es_laptop_baja else 16), sticky="ew")

    lbl_subtitulo = tk.Label(
        panel,
        text="Se detecto un nivel ROSA >= 5.\nCorrige la postura antes de continuar.",
        font=("Consolas", subtitle_font),
        bg="#120000",
        fg="white",
        justify="center",
        wraplength=subtitle_wrap,
    )
    lbl_subtitulo.grid(row=1, column=0, padx=pad_x, pady=(0, 14 if es_laptop_baja else 18), sticky="ew")

    area_lista = tk.Frame(panel, bg="#120000")
    area_lista.grid(row=2, column=0, padx=pad_x, pady=(0, 12 if es_laptop_baja else 18), sticky="nsew")
    area_lista.grid_columnconfigure(0, weight=1)
    area_lista.grid_rowconfigure(0, weight=1)

    canvas = tk.Canvas(
        area_lista,
        bg="#120000",
        highlightthickness=0,
        bd=0,
        relief="flat",
    )
    scrollbar = tk.Scrollbar(area_lista, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns", padx=(8, 0))

    contenedor = tk.Frame(canvas, bg="#120000")
    canvas_window = canvas.create_window((0, 0), window=contenedor, anchor="nw")

    def _actualizar_scroll(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfigure(canvas_window, width=max(canvas.winfo_width(), 1))

    contenedor.bind("<Configure>", _actualizar_scroll)
    canvas.bind("<Configure>", _actualizar_scroll)

    for item in recomendaciones:
        fila = tk.Frame(contenedor, bg="#1A0505", highlightthickness=1, highlightbackground="#5F2222")
        fila.pack(fill="x", pady=row_gap)

        if item.get("icono"):
            icono = _cargar_icono_categoria(item["icono"], size=icon_size)
            overlay._iconos_alerta.append(icono)
            tk.Label(fila, image=icono, bg="#1A0505").pack(side="left", padx=(12, 10), pady=row_pad)

        cuerpo = tk.Frame(fila, bg="#1A0505")
        cuerpo.pack(side="left", fill="both", expand=True, padx=(0, 12), pady=row_pad)

        tk.Label(
            cuerpo,
            text=f"{item['categoria']} (puntaje {item['score']}):",
            font=("Consolas", section_font, "bold"),
            bg="#1A0505",
            fg="#FFE082",
            justify="left",
            anchor="w",
            wraplength=action_wrap,
        ).pack(anchor="w")

        for accion in item.get("acciones", []):
            tk.Label(
                cuerpo,
                text=f"- {accion}",
                font=("Consolas", action_font, "bold"),
                bg="#1A0505",
                fg="#F8EFD0",
                justify="left",
                anchor="w",
                wraplength=action_wrap,
            ).pack(anchor="w", pady=(2 if es_laptop_baja else 3, 0))

    lbl_footer = tk.Label(
        panel,
        text="Haz clic o presiona Escape para desbloquear.",
        font=("Consolas", footer_font, "bold"),
        bg="#120000",
        fg="#FF8A80",
        wraplength=subtitle_wrap,
    )
    lbl_footer.grid(row=3, column=0, padx=pad_x, pady=(0, bottom_pad), sticky="ew")

    def _on_mousewheel(event):
        if canvas.winfo_height() < contenedor.winfo_reqheight():
            delta = -1 if event.delta < 0 else 1
            canvas.yview_scroll(-delta, "units")

    canvas.bind("<MouseWheel>", _on_mousewheel)
    contenedor.bind("<MouseWheel>", _on_mousewheel)

    def cerrar_alerta(_event=None):
        try:
            overlay.grab_release()
        except Exception:
            pass
        if callable(on_close):
            on_close()
        overlay.destroy()

    overlay.bind("<Button-1>", cerrar_alerta)
    overlay.bind("<Escape>", cerrar_alerta)

    threading.Thread(
        target=_reproducir_sonido_alerta,
        daemon=True,
    ).start()

    return overlay


class PanelROSA(tk.Tk):
    def __init__(self, nombre, horas_diarias, cam_idx, ruta_xlsx, config_ergonomica):
        super().__init__()
        self.title(f"Proyecto de Titulación - ROSA ERGONOMY - CESAR  | {nombre}")
        self.geometry("1120x660")
        self.minsize(1080, 620)
        self.configure(bg=C["bg_deep"])

        self.cola_datos = queue.Queue()
        self.cola_frames = queue.Queue(maxsize=2)
        self.cola_angulos = queue.Queue(maxsize=2)
        self.registros = []
        self.ruta_xlsx = ruta_xlsx
        self.num_registro = 0
        self._ultimo_guardado = time.time()
        self.cam_idx = cam_idx
        self.cfg = config_ergonomica
        self._alerta_overlay = None
        self._alerta_activa = False
        self._ultimo_alerta = 0.0
        self._after_update_id = None
        self._after_alarma_id = None
        self._cerrando = False
        self._guardado_lock = threading.Lock()
        self._guardado_en_progreso = False
        self._guardado_pendiente = None
        self.cola_estado_guardado = queue.Queue()
        self._alarma_pendiente = False
        self._alarma_deadline = 0.0
        self._ultimo_dato = None

        self.metadata = {
            "nombre_proyecto": self.cfg.get("nombre_proyecto", "Evaluacion ergonomica ROSA"),
            "evaluador": self.cfg.get("evaluador", "Jaime Cesar Tarazona Tinoco"),
            "empresa": self.cfg.get("empresa", "CHYC Ingenieros SAC"),
            "trabajador": nombre,
            "horas_diarias": horas_diarias,
            "camara": f"Camara {cam_idx}",
        }

        crear_excel_nuevo(self.ruta_xlsx, metadata=self.metadata, cfg_inicial=self.cfg)
        self._setup_ui(nombre, horas_diarias)

        self.hilo = HiloCamara(
            self.cola_datos,
            self.cola_frames,
            self.cola_angulos,
            horas_diarias,
            cam_idx,
            config_ergonomica,
        )
        self.hilo.start()
        self._update_loop()

    def _setup_ui(self, nombre, horas_diarias):
        left = tk.Frame(self, bg=C["bg_deep"])
        left.pack(side="left", fill="both", expand=True, padx=(16, 8), pady=16)

        self.canvas = tk.Canvas(
            left,
            width=640,
            height=480,
            bg="black",
            highlightthickness=1,
            highlightbackground=C["bg_border"],
        )
        self.canvas.pack(fill="both", expand=True)
        self.img_id = self.canvas.create_image(0, 0, anchor="nw")

        info_bar = tk.Frame(left, bg=C["bg_panel"])
        info_bar.pack(fill="x", pady=(8, 0))
        self.lbl_estado_camara = tk.Label(
            info_bar,
            text="Inicializando camara...",
            font=("Consolas", 9),
            bg=C["bg_panel"],
            fg=C["text_lo"],
        )
        self.lbl_estado_camara.pack(side="left", padx=8, pady=6)
        self.lbl_xlsx_status = tk.Label(
            info_bar,
            text="",
            font=("Consolas", 9),
            bg=C["bg_panel"],
            fg=C["ok"],
        )
        self.lbl_xlsx_status.pack(side="right", padx=8, pady=6)
        self.lbl_alerta_status = tk.Label(
            info_bar,
            text="",
            font=("Consolas", 9, "bold"),
            bg=C["bg_panel"],
            fg=C["warn"],
        )
        self.lbl_alerta_status.pack(side="right", padx=(0, 8), pady=6)

        right = tk.Frame(self, bg=C["bg_deep"], width=420)
        right.pack(side="right", fill="y", padx=(8, 16), pady=16)
        right.pack_propagate(False)

        right_content = tk.Frame(right, bg=C["bg_deep"])
        right_content.pack(fill="both", expand=True)

        right_footer = tk.Frame(right, bg=C["bg_deep"])
        right_footer.pack(fill="x", side="bottom")

        self._card_header(right_content, "Resultado")
        score_card = self._card(right_content)
        self.lbl_score = tk.Label(
            score_card,
            text="--",
            font=("Consolas", 58, "bold"),
            bg=C["bg_card"],
            fg=C["accent"],
        )
        self.lbl_score.pack(pady=(10, 2))
        self.lbl_nivel = tk.Label(
            score_card,
            text="EN ESPERA",
            font=("Consolas", 12, "bold"),
            bg=C["bg_card"],
            fg=C["text_mid"],
        )
        self.lbl_nivel.pack(pady=(0, 10))

        self._card_header(right_content, "Resumen actual")
        resumen_card = self._card(right_content)
        self.lbl_resumen = tk.Label(
            resumen_card,
            text=f"Esperando primera medicion ({INTERVALO_EVAL_SEG}s)...",
            font=("Consolas", 9),
            bg=C["bg_card"],
            fg=C["text_hi"],
            justify="left",
            anchor="w",
            wraplength=370,
        )
        self.lbl_resumen.pack(fill="x", padx=12, pady=12)

        self._card_header(right_content, "Desglose ROSA")
        desglose_card = self._card(right_content)
        self.lbl_desglose = tk.Label(
            desglose_card,
            text="Sin datos todavía.",
            font=("Consolas", 9),
            bg=C["bg_card"],
            fg=C["text_hi"],
            justify="left",
            anchor="w",
            wraplength=370,
        )
        self.lbl_desglose.pack(fill="x", padx=12, pady=12)


        self._card_header(right_content, "Sesion")
        session_card = self._card(right_content)
        self.lbl_sesion = tk.Label(
            session_card,
            text=(
                f"Trabajador: {nombre}\n"
                f"Proyecto: {self.metadata['nombre_proyecto']}\n"
                f"Evaluador: {self.metadata['evaluador']}\n"
                f"Empresa: {self.metadata['empresa']}\n"
                f"Horas/dia: {horas_diarias}\n"
                f"Fuente: Camara {self.cam_idx}\n"
                f"Evaluacion ROSA: cada {INTERVALO_EVAL_SEG}s\n"
                f"Excel: {self.ruta_xlsx}\n"
                f"Auto-guardado: {GUARDADO_SEG}s"
            ),
            font=("Consolas", 8),
            bg=C["bg_card"],
            fg=C["text_lo"],
            justify="left",
            anchor="w",
            wraplength=370,
        )
        self.lbl_sesion.pack(fill="x", padx=12, pady=12)

        tk.Button(
            right_footer,
            text="GUARDAR EXCEL",
            font=("Consolas", 10, "bold"),
            bg=C["accent2"],
            fg="white",
            relief="flat",
            cursor="hand2",
            command=self._guardar_xlsx_manual,
        ).pack(fill="x", pady=(12, 0), ipady=7)


    def _card_header(self, parent, text):
        tk.Label(
            parent,
            text=text.upper(),
            font=("Consolas", 9, "bold"),
            bg=C["bg_deep"],
            fg=C["accent"],
            anchor="w",
        ).pack(fill="x", pady=(0, 4))

    def _card(self, parent):
        card = tk.Frame(
            parent,
            bg=C["bg_card"],
            highlightthickness=1,
            highlightbackground=C["bg_border"],
        )
        card.pack(fill="x", pady=(0, 10))
        return card

    def _actualizar_frame(self, frame_rgb):
        img = Image.fromarray(frame_rgb).resize((640, 480))
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.config(width=img.width, height=img.height)
        self.canvas.itemconfig(self.img_id, image=imgtk)
        self.canvas.image = imgtk
        self.lbl_estado_camara.config(text="Camara activa")

    def _actualizar_resultado(self, dato):
        score = dato["rosa"]
        texto_nivel, color = nivel_accion(score)
        self.lbl_score.config(text=str(score), fg=color)
        self.lbl_nivel.config(text=texto_nivel, fg=color)
        cfg_eval = cfg_efectiva(self.cfg, dato.get("flags_inferidos"))
        contexto_obj = dato.get("contexto_objetos")
        contexto_linea = f"\nObjetos: {contexto_obj}" if contexto_obj else ""
        if contexto_obj and not dato.get("contexto_completo", True):
            contexto_linea += "\nContexto visual incompleto"
        motivos_inferidos = dato.get("motivos_inferidos", "")
        if motivos_inferidos:
            contexto_linea += f"\nInferido por camara: {motivos_inferidos}"
        self.lbl_resumen.config(
            text=(
                f"Silla total: {dato['total_silla']}\n"
                f"Pantalla y telefono: {dato['tabla_B']}\n"
                f"Mouse y teclado: {dato['tabla_C']}\n"
                f"Combinado perifericos: {dato['tabla_D']}\n"
                f"Factor tiempo diario aplicado: {dato['factor_tiempo_silla']:+d}\n"
                f"Ultima medicion: {dato['time']}"
                f"{contexto_linea}"
            )
        )
        self.lbl_desglose.config(
            text=(
                f"A1 Asiento altura: {dato['A1']} | {motivo_a1(dato, cfg_eval)}\n"
                f"A2 Asiento profundidad: {dato['A2']} | {motivo_a2(dato, cfg_eval)}\n"
                f"A3 Reposabrazos: {dato['A3']} | {motivo_a3(dato, cfg_eval)}\n"
                f"A4 Respaldo: {dato['A4']} | {motivo_a4(dato, cfg_eval)}\n"
                f"Tabla A: {dato['tabla_A']}  Tiempo diario: {dato['factor_tiempo_silla']:+d}\n"
                f"B1 Telefono: {dato['B1']} | {motivo_b1(dato, cfg_eval)}\n"
                f"B2 Pantalla: {dato['B2']} | {motivo_b2(dato, cfg_eval)}\n"
                f"Tabla B: {dato['tabla_B']}  Tiempo diario: {dato['factor_tiempo_pantalla']:+d}\n"
                f"C1 Mouse: {dato['C1']} | {motivo_c1(dato, cfg_eval)}\n"
                f"C2 Teclado: {dato['C2']} | {motivo_c2(dato, cfg_eval)}\n"
                f"Tabla C: {dato['tabla_C']}  Tiempo diario: {dato['factor_tiempo_teclado']:+d}\n"
                f"Tabla D: {dato['tabla_D']}\n"
                f"ROSA final = max(Silla total, Tabla D)"
            )
        )

    def _update_loop(self):
        while True:
            try:
                estado_guardado = self.cola_estado_guardado.get_nowait()
            except queue.Empty:
                break

            if estado_guardado["ok"]:
                self._ultimo_guardado = estado_guardado["ts"]
                self.lbl_xlsx_status.config(
                    text=f"Guardado {time.strftime('%H:%M:%S', time.localtime(estado_guardado['ts']))}",
                    fg=C["ok"],
                )
            else:
                self.lbl_xlsx_status.config(text=f"Error: {estado_guardado['error']}", fg=C["danger"])

        frame_rgb = None
        try:
            while True:
                frame_rgb = self.cola_frames.get_nowait()
        except queue.Empty:
            pass
        if frame_rgb is not None:
            self._actualizar_frame(frame_rgb)

        try:
            while True:
                self.cola_angulos.get_nowait()
        except queue.Empty:
            pass

        dato = None
        try:
            while True:
                dato = self.cola_datos.get_nowait()
        except queue.Empty:
            pass
        if dato is not None:
            if "error" in dato:
                messagebox.showerror("Error", dato["error"])
                self.destroy()
                return
            self._ultimo_dato = dato
            self.registros.append(dato)
            self.num_registro += 1
            self._actualizar_resultado(dato)
            if dato["rosa"] >= NIVEL_ACCION:
                self._programar_alarma()
            else:
                self._cancelar_alarma_pendiente()

        if self._alarma_pendiente:
            restante = max(0, int(self._alarma_deadline - time.time() + 0.999))
            self.lbl_alerta_status.config(text=f"Alarma en {restante}s")
        elif not self._alerta_activa:
            self.lbl_alerta_status.config(text="")

        if self.registros and time.time() - self._ultimo_guardado >= GUARDADO_SEG:
            self._programar_guardado_async()

        if not self._cerrando:
            self._after_update_id = self.after(30, self._update_loop)

    def _programar_guardado_async(self):
        if not self.registros or self._cerrando:
            return

        payload = (
            copy.deepcopy(self.registros[-1]),
            self.num_registro,
        )

        with self._guardado_lock:
            self._guardado_pendiente = payload
            if self._guardado_en_progreso:
                return
            self._guardado_en_progreso = True
            payload = self._guardado_pendiente
            self._guardado_pendiente = None

        threading.Thread(
            target=self._guardar_worker,
            args=(payload,),
            daemon=True,
        ).start()

    def _guardar_worker(self, payload_inicial):
        payload = payload_inicial
        while payload is not None:
            registro, num_registro = payload
            try:
                agregar_registro_excel(
                    self.ruta_xlsx,
                    registro,
                    num_registro,
                    metadata=self.metadata,
                    cfg_base=self.cfg,
                )
                self.cola_estado_guardado.put({"ok": True, "ts": time.time()})
            except Exception as exc:
                self.cola_estado_guardado.put({"ok": False, "error": str(exc)})

            with self._guardado_lock:
                payload = self._guardado_pendiente
                self._guardado_pendiente = None
                if payload is None:
                    self._guardado_en_progreso = False
                    break

    def _guardar_nuevo_registro(self):
        if not self.registros:
            return
        try:
            agregar_registro_excel(
                self.ruta_xlsx,
                self.registros[-1],
                self.num_registro,
                metadata=self.metadata,
                cfg_base=self.cfg,
            )
            self._ultimo_guardado = time.time()
            self.lbl_xlsx_status.config(
                text=f"Guardado {time.strftime('%H:%M:%S')}",
                fg=C["ok"],
            )
        except Exception as exc:
            self.lbl_xlsx_status.config(text=f"Error: {exc}", fg=C["danger"])

    def _guardar_xlsx_manual(self):
        ruta = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel", "*.xlsx")],
            title="Guardar evaluacion ROSA",
            initialfile=f"ROSA_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
        )
        if not ruta:
            return
        try:
            crear_excel_nuevo(ruta, metadata=self.metadata, cfg_inicial=self.cfg)
            for i, registro in enumerate(self.registros, 1):
                agregar_registro_excel(ruta, registro, i, metadata=self.metadata, cfg_base=self.cfg)
            messagebox.showinfo("Guardado", f"Excel guardado en:\n{ruta}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _cancelar_alarma_pendiente(self):
        self._alarma_pendiente = False
        self._alarma_deadline = 0.0
        self.lbl_alerta_status.config(text="")
        if self._after_alarma_id is not None:
            try:
                self.after_cancel(self._after_alarma_id)
            except Exception:
                pass
            self._after_alarma_id = None

    def _programar_alarma(self):
        ahora = time.time()
        if self._alerta_activa or self._alarma_pendiente or (ahora - self._ultimo_alerta) < 3:
            return
        self._alarma_pendiente = True
        self._alarma_deadline = ahora + RETARDO_ALARMA_SEG
        self.lbl_alerta_status.config(text=f"Alarma en {RETARDO_ALARMA_SEG}s")
        self._after_alarma_id = self.after(RETARDO_ALARMA_SEG * 1000, self._mostrar_alarma_diferida)

    def _mostrar_alarma_diferida(self):
        self._after_alarma_id = None
        self._alarma_pendiente = False
        self._alarma_deadline = 0.0
        self.lbl_alerta_status.config(text="")
        ultimo = self._ultimo_dato or (self.registros[-1] if self.registros else {})
        if not ultimo or ultimo.get("rosa", 0) < NIVEL_ACCION or self._cerrando:
            return
        self._alertar(ultimo)

    def _alertar(self, dato=None):
        ahora = time.time()
        if self._alerta_activa or (ahora - self._ultimo_alerta) < 3:
            return

        self._ultimo_alerta = ahora
        self._alerta_activa = True

        def cerrar_alerta(_event=None):
            if not self._alerta_activa:
                return
            self._alerta_activa = False
            self._alerta_overlay = None
            self.lbl_alerta_status.config(text="")
        ultima = dato or self._ultimo_dato or (self.registros[-1] if self.registros else {})
        recomendaciones = recomendaciones_alerta(ultima, cfg_efectiva(self.cfg, ultima.get("flags_inferidos")))
        self._alerta_overlay = mostrar_overlay_alarma(self, recomendaciones, on_close=cerrar_alerta)

    def on_close(self):
        if self._cerrando:
            return
        self._cerrando = True

        if self._after_update_id is not None:
            try:
                self.after_cancel(self._after_update_id)
            except Exception:
                pass
            self._after_update_id = None

        self._cancelar_alarma_pendiente()

        if self._alerta_overlay is not None:
            try:
                self._alerta_overlay.grab_release()
            except Exception:
                pass
            try:
                self._alerta_overlay.destroy()
            except Exception:
                pass
            self._alerta_overlay = None

        self.hilo.stop()
        if self.hilo.is_alive():
            self.hilo.join(timeout=2.0)

        self.destroy()
