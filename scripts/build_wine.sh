#!/usr/bin/env bash
# ============================================================
# build_wine.sh — Compila EvaluadorROSA.exe usando Wine Python
# Prerequisito: bash scripts/setup_wine.sh
# Uso: bash scripts/build_wine.sh
# ============================================================
set -e

export WINEPREFIX="${WINEPREFIX:-$HOME/.wine_rosa}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== EvaluadorROSA — Build para Windows (Wine) ==="
echo "Directorio del proyecto: $PROJECT_ROOT"

# Localizar python.exe dentro del prefijo Wine
WINE_PYTHON=$(find "$WINEPREFIX" -name "python.exe" 2>/dev/null | grep -v "_pth" | head -1)
if [ -z "$WINE_PYTHON" ]; then
    echo "ERROR: Python no encontrado en $WINEPREFIX"
    echo "Ejecuta primero: bash scripts/setup_wine.sh"
    exit 1
fi

echo "Wine Python: $WINE_PYTHON"
WINE_PYTHON_VERSION=$(WINEPREFIX="$WINEPREFIX" wine "$WINE_PYTHON" --version 2>&1)
echo "Versión: $WINE_PYTHON_VERSION"

# Instalar dependencias en Wine Python
echo ""
echo "[1/3] Instalando dependencias en Wine Python..."
WINEPREFIX="$WINEPREFIX" wine "$WINE_PYTHON" -m pip install --upgrade pip
WINEPREFIX="$WINEPREFIX" wine "$WINE_PYTHON" -m pip install \
    opencv-python mediapipe numpy openpyxl Pillow pyinstaller

# Verificar que PyInstaller quedó instalado
WINE_PYINSTALLER=$(find "$WINEPREFIX" -name "pyinstaller.exe" 2>/dev/null | head -1)
if [ -z "$WINE_PYINSTALLER" ]; then
    echo "ERROR: PyInstaller no encontrado tras la instalación."
    exit 1
fi

# Compilar
echo ""
echo "[2/3] Compilando EvaluadorROSA.exe con PyInstaller..."
cd "$PROJECT_ROOT"
WINEPREFIX="$WINEPREFIX" wine "$WINE_PYTHON" -m PyInstaller \
    --noconfirm \
    --clean \
    EvaluadorROSA.spec

# Verificar salida
EXE="$PROJECT_ROOT/dist/EvaluadorROSA.exe"
if [ -f "$EXE" ]; then
    SIZE=$(du -sh "$EXE" | cut -f1)
    echo ""
    echo "[3/3] Build completado:"
    echo "  Ejecutable: $EXE"
    echo "  Tamaño:     $SIZE"
    echo ""
    echo "Para probar: bash scripts/run_wine.sh"
else
    echo "ERROR: No se generó el ejecutable en dist/EvaluadorROSA.exe"
    exit 1
fi
