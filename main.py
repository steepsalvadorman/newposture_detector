import sys

from gui_inicio import DialogoInicio
from gui_principal import PanelROSA
from tablas_calculos import MODEL_PATH


if __name__ == "__main__":
    print("[main] Iniciando Evaluador ROSA NTP-1173 v5.0")
    print(f"[main] Modelo: {MODEL_PATH}")

    dialogo = DialogoInicio()
    dialogo.mainloop()

    if dialogo.resultado is None:
        print("[main] Cancelado por el usuario.")
        sys.exit(0)

    nombre, horas, cam_idx, ruta_xlsx, cfg_ergonomica = dialogo.resultado
    print(f"[main] Trabajador: {nombre} | Horas: {horas} | Cam: {cam_idx}")
    print(f"[main] Config: {cfg_ergonomica}")

    app = PanelROSA(nombre, horas, cam_idx, ruta_xlsx, cfg_ergonomica)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
    print("[main] Aplicación cerrada.")
