#!/usr/bin/env bash
# ============================================================
# run_wine.sh — Ejecuta EvaluadorROSA.exe en Wine para pruebas
# Uso: bash scripts/run_wine.sh [--debug]
# ============================================================

export WINEPREFIX="${WINEPREFIX:-$HOME/.wine_rosa}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXE="$PROJECT_ROOT/dist/EvaluadorROSA.exe"

if [ ! -f "$EXE" ]; then
    echo "ERROR: $EXE no encontrado."
    echo "Ejecuta primero: bash scripts/build_wine.sh"
    exit 1
fi

if ! command -v wine &>/dev/null; then
    echo "ERROR: Wine no instalado. Ejecuta: bash scripts/setup_wine.sh"
    exit 1
fi

# Redirigir logs de Wine a archivo para no contaminar la consola
WINE_LOG="$PROJECT_ROOT/dist/wine_run.log"

echo "=== Ejecutando EvaluadorROSA.exe en Wine ==="
echo "Exe:  $EXE"
echo "Log:  $WINE_LOG"
echo ""

if [[ "$1" == "--debug" ]]; then
    WINEDEBUG="+all" WINEPREFIX="$WINEPREFIX" wine "$EXE" 2>&1 | tee "$WINE_LOG"
else
    WINEDEBUG="-all" WINEPREFIX="$WINEPREFIX" wine "$EXE" 2>"$WINE_LOG"
fi
