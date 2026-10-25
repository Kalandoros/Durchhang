import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Durchhang — Berechnungsprotokoll")


@app.cell
def _():
    import marimo as mo
    import json
    import math
    return json, math, mo


@app.cell
def _(json, math):
    import os
    from pathlib import Path
    # Im interaktiven marimo-Modus ist __file__ nicht immer korrekt.
    try:
        _state_file = Path(__file__).parent / "state.json"
        if not _state_file.exists():
            raise FileNotFoundError
    except (NameError, FileNotFoundError):
        _state_file = Path(os.getcwd()) / "src" / "state.json"
    with open(_state_file, "r", encoding="utf-8") as f:
        state = json.load(f)
    F_G = state["F_G"]
    F_W = state["F_W"]
    sigma_0 = state["sigma_0"]
    span_L = state["span_L"]
    E_modul = state["E_modul"]
    A_cross = state["A_cross"]
    n_sub = state["n_sub"]
    mode = state["mode"]
    calc_method = state["calc_method"]
    f_0_out = state.get("f_0_out")
    sigma_1_out = state.get("sigma_1_out")
    ausschwingwinkel_out = state.get("ausschwingwinkel_out")
    return A_cross, E_modul, F_G, F_W, ausschwingwinkel_out, calc_method, f_0_out, mode, n_sub, sigma_0, sigma_1_out, span_L, state


@app.cell
def _(mo):
    mo.md("""
# Durchhang — Berechnungsprotokoll

Starte zuerst die Taipy-App (`Windeinfluss_mit_exakt.py`), führe im Browser eine Berechnung durch.
Die Werte werden automatisch aus `state.json` gelesen.
""")
    return


@app.cell
def _(F_G, F_W, sigma_0, span_L):
    print(f"F_G = {F_G} N/m,  F_W = {F_W} N/m")
    print(f"sigma_0 = {sigma_0} N/mm²,  L = {span_L} m")
    return


@app.cell
def _(mo):
    mo.md("""
## 1. Windeinfluss

Die resultierende Last $F_{ges}$ aus Eigengewicht und Windlast nach Pythagoras.
""")
    return


@app.cell
def _(F_G, F_W, math, mo):
    F_ges = math.sqrt(F_G**2 + F_W**2)
    mo.md(f"""
$$F_{{ges}} = \\sqrt{{F_{{G}}^2 + F_{{W}}^2}} = \\sqrt{{{F_G}^2 + {F_W}^2}} = {F_ges:.3f} \\; \\text{{N/m}}$$
""")
    return F_ges


@app.cell
def _(mo):
    mo.md("""
## 2. Ausschwingwinkel

Der Winkel $\\alpha$, um den der Leiter durch den Wind ausgeschlagen wird.
""")
    return


@app.cell
def _(F_G, F_W, math, mo):
    _alpha = math.degrees(math.atan(F_G / F_W))
    mo.md(f"""
$$\\alpha = \\arctan\\left(\\frac{{F_{{G}}}}{{F_{{W}}}}\\right) = \\arctan\\left(\\frac{{{F_G}}}{{{F_W}}}\\right) = {_alpha:.2f}^\\circ$$
""")
    return _alpha


@app.cell
def _(mo):
    mo.md("""
## Werte aktualisieren

Wenn du Werte in der Taipy-GUI (im Browser) änderst, klicke den Button unten,
um die neuen Werte aus `state.json` zu lesen. Alle Formeln werden automatisch neu berechnet.
""")
    return


@app.cell
def _(mo):
    btn = mo.ui.refresh()
    return btn


if __name__ == "__main__":
    app.run()
