#!/usr/bin/env bash
# ============================================================
# setup_wine.sh — Instala Wine + Python 3.12 Windows en Wine
# Uso: bash scripts/setup_wine.sh
# ============================================================
set -e

PYTHON_VERSION="3.12.10"
PYTHON_INSTALLER="python-${PYTHON_VERSION}-amd64.exe"
PYTHON_URL="https://www.python.org/ftp/python/${PYTHON_VERSION}/${PYTHON_INSTALLER}"

echo "=== EvaluadorROSA — Configuración Wine ==="

# 1. Instalar Wine si no está
if ! command -v wine &>/dev/null; then
    echo "[1/4] Instalando Wine (necesita sudo)..."
    sudo pacman -S --noconfirm wine wine-gecko wine-mono winetricks
else
    echo "[1/4] Wine ya instalado: $(wine --version)"
fi

# 2. Configurar Wine para 64-bit
export WINEARCH=win64
export WINEPREFIX="$HOME/.wine_rosa"
if [ ! -d "$WINEPREFIX" ]; then
    echo "[2/4] Inicializando prefijo Wine en $WINEPREFIX ..."
    wineboot --init
else
    echo "[2/4] Prefijo Wine ya existe: $WINEPREFIX"
fi

# 3. Instalar Python para Windows en Wine
WINE_PYTHON=$(find "$WINEPREFIX" -name "python.exe" 2>/dev/null | head -1)
if [ -z "$WINE_PYTHON" ]; then
    echo "[3/4] Descargando Python $PYTHON_VERSION para Windows..."
    TMP_DIR=$(mktemp -d)
    curl -L "$PYTHON_URL" -o "$TMP_DIR/$PYTHON_INSTALLER"
    echo "Instalando Python en Wine (sigue las instrucciones del instalador)..."
    echo "IMPORTANTE: marca 'Add python.exe to PATH' antes de continuar."
    WINEPREFIX="$WINEPREFIX" wine "$TMP_DIR/$PYTHON_INSTALLER"
    rm -rf "$TMP_DIR"
else
    echo "[3/4] Python ya instalado en Wine: $WINE_PYTHON"
fi

# 4. Verificar la instalación
WINE_PYTHON=$(find "$WINEPREFIX" -name "python.exe" 2>/dev/null | head -1)
if [ -z "$WINE_PYTHON" ]; then
    echo "ERROR: Python no encontrado en Wine. Vuelve a ejecutar el instalador."
    exit 1
fi

echo "[4/4] Verificando Python en Wine..."
WINEPREFIX="$WINEPREFIX" wine "$WINE_PYTHON" --version

echo ""
echo "=== Setup completado ==="
echo "Wine Python: $WINE_PYTHON"
echo "Siguiente paso: bash scripts/build_wine.sh"
