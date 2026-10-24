from taipy.gui import Gui
import taipy.gui.builder as tgb
import math
import pandas as pd


# --- LOGIK & BERECHNUNG ---
def berechne_eislast(A_mm2: float, d_eis_mm: float, rho_eis: float) -> float:
    """Berechnet die Eislast F_Eis in N/m basierend auf Seildurchmesser und Eisschichtdicke."""
    if d_eis_mm <= 0 or A_mm2 <= 0 or rho_eis <= 0:
        return 0.0

    d_seil_mm = math.sqrt((4.0 * A_mm2) / math.pi)
    A_eis_mm2 = math.pi * d_eis_mm * (d_seil_mm + d_eis_mm)
    A_eis_m2 = A_eis_mm2 * 1e-6

    F_eis_N_m = A_eis_m2 * rho_eis * 9.81
    return F_eis_N_m


def berechne_windeinfluss(F_G: float, F_W: float) -> float:
    return math.sqrt(F_G**2 + F_W**2)


def berechne_ausschwingwinkel(F_G: float, F_W: float) -> float:
    if F_W <= 0:
        return 90.0
    return math.degrees(math.atan(F_G / F_W))


def berechne_durchhang_max(F_G: float, l: float, H_N: float) -> float:
    if H_N <= 0:
        return 0.0
    return ((l**2 * F_G) / (8.0 * H_N)) + ((l**4 * F_G**3) / (384.0 * H_N**3))


def berechne_neue_seilspannung(sigma_0_N_mm2: float, F_G_init_N_m: float, F_Load_N_m: float, E_N_mm2: float, L_m: float, A_mm2: float, alpha_th: float, delta_T: float) -> tuple[float, float]:
    if sigma_0_N_mm2 <= 0 or E_N_mm2 <= 0 or L_m <= 0 or A_mm2 <= 0:
        raise ValueError("Spannung, E-Modul, Querschnitt und Spannweite müssen größer als 0 sein.")

    sigma_0 = sigma_0_N_mm2 * 1e6  # N/mm² -> Pa
    E = E_N_mm2 * 1e6  # N/mm² -> Pa
    A = A_mm2 * 1e-6  # mm² -> m²
    L = L_m  # m

    w_0 = F_G_init_N_m  # N/m
    w_1 = F_Load_N_m  # N/m

    K_0 = (E * (L**2) * (w_0**2)) / (24.0 * (A**2))
    K_1 = (E * (L**2) * (w_1**2)) / (24.0 * (A**2))

    RHS = sigma_0 - (K_0 / (sigma_0**2)) - (E * alpha_th * delta_T)

    def f(sig_Pa):
        if sig_Pa <= 0:
            return -float('inf')
        return sig_Pa - (K_1 / (sig_Pa**2)) - RHS

    low = 1.0
    high = 1e11

    val_low = f(low)
    val_high = f(high)

    if val_low * val_high > 0:
        raise RuntimeError("Gleichgewichtslage konnte für diese Parameter nicht berechnet werden.")

    for _ in range(100):
        mid = (low + high) / 2.0
        val = f(mid)
        if abs(val) < 1e-2:
            break
        if val * val_low < 0:
            high = mid
            val_high = val
        else:
            low = mid
            val_low = val

    sigma_1_Pa = (low + high) / 2.0
    sigma_1_N_mm2 = sigma_1_Pa / 1e6
    H_zug_kN = (sigma_1_Pa * A) / 1000.0

    return sigma_1_N_mm2, H_zug_kN


ALPHA_TH = 0.000023


def berechne_einzelnen_modus(m_name: str, p: dict) -> dict:
    has_wind = "Wind" in m_name
    has_eis = "Eis" in m_name

    if m_name == "Temperatur -20°C":
        T_end = -20.0
    elif m_name == "Temperatur -5°C + Eis":
        T_end = -5.0
    elif m_name == "Temperatur +5°C + Wind":
        T_end = 5.0
    elif m_name == "Temperatur 10°C":
        T_end = 10.0
    elif m_name == "Temperatur 80°C":
        T_end = 80.0
    else:
        T_end = 15.0

    f_eis = berechne_eislast(p['A_cross'], float(p['d_eis']), p['rho_eis']) if has_eis else 0.0
    f_g_ges = p['F_G'] + f_eis
    w_quer = p['F_W'] if has_wind else 0.0
    load_force = berechne_windeinfluss(f_g_ges, p['F_W']) if has_wind else f_g_ges
    delta_T = T_end - p['T_start'] if "Temperatur" in m_name else 0.0

    try:
        sigma_1, H_kN = berechne_neue_seilspannung(sigma_0_N_mm2=p['sigma_0'], F_G_init_N_m=p['F_G'], F_Load_N_m=load_force, E_N_mm2=p['E_modul'], L_m=p['span_L'], A_mm2=p['A_cross'],
            alpha_th=ALPHA_TH, delta_T=delta_T)
        V_kN = (f_g_ges * p['span_L'] / 2.0) / 1000.0
        W_trans_kN = (w_quer * p['span_L'] / 2.0) / 1000.0
        F_anker_kN = math.sqrt(H_kN**2 + V_kN**2 + W_trans_kN**2)
        H_1_N = H_kN * 1000.0
        f_1 = berechne_durchhang_max(F_G=load_force, l=p['span_L'], H_N=H_1_N)

        return {"sigma_1":round(sigma_1, 3), "H_zug":round(H_kN, 3), "V_kraft":round(V_kN, 3), "W_kraft":round(W_trans_kN, 3), "F_anker":round(F_anker_kN, 3), "f_1":round(f_1, 3)}
    except Exception:
        return {"sigma_1":0.0, "H_zug":0.0, "V_kraft":0.0, "W_kraft":0.0, "F_anker":0.0, "f_1":0.0}


def erstelle_uebersichtstabelle(p: dict) -> pd.DataFrame:
    spalten_modi = ["Temperatur -20°C", "Temperatur -5°C + Eis", "Temperatur +5°C + Wind", "Temperatur 10°C", "Temperatur 80°C"]

    results = {m:berechne_einzelnen_modus(m, p) for m in spalten_modi}

    zeilen_definition = [("Finale Seilspannung σ₁ [N/mm²]", "sigma_1"), ("Horizontale Seilzugkraft H [kN]", "H_zug"), ("Vertikale Auflagerkraft V [kN]", "V_kraft"),
        ("Resultierende Gesamtzugkraft F_Anker [kN]", "F_anker"), ("Finaler Seildurchhang f₁ [m]", "f_1"), ("Transversale Windauflagerkraft W [kN]", "W_kraft")]

    data = {"Kenngröße / Parameter":[z[0] for z in zeilen_definition]}
    for m in spalten_modi:
        data[m] = [results[m][key] for _, key in zeilen_definition]

    return pd.DataFrame(data)


# --- STATE INITIALISIERUNG ---
mode = "Wind"
F_G = 10.849
F_W = 24.544
F_Z = 26.920
T_start = 10.0
T_end = -20.0
sigma_0 = 12.5
E_modul = 55000.0
span_L = 20.0
A_cross = 400.0

rho_eis = 900.0
d_eis = 15

f_0_out = 0.108
sigma_1_out = 19.225
H_zug_out = 7.690
V_kraft_out = 0.108
W_kraft_out = 0.245
F_anker_out = 7.695
f_1_out = 0.175
ausschwingwinkel_out = 23.86
error_msg = ""

init_params = {'F_G':F_G, 'F_W':F_W, 'T_start':T_start, 'sigma_0':sigma_0, 'E_modul':E_modul, 'span_L':span_L, 'A_cross':A_cross, 'rho_eis':rho_eis, 'd_eis':d_eis}
summary_df = erstelle_uebersichtstabelle(init_params)


def on_change(state, var_name, var_value):
    state.error_msg = ""

    if var_name == "mode":
        if var_value == "Temperatur +5°C + Wind":
            state.T_end = 5.0
        elif var_value == "Temperatur -20°C":
            state.T_end = -20.0
        elif var_value == "Temperatur -5°C + Eis":
            state.T_end = -5.0
        elif var_value == "Temperatur 10°C":
            state.T_end = 10.0
        elif var_value == "Temperatur 80°C":
            state.T_end = 80.0
        elif var_value == "Temperatur":
            state.T_end = 15.0

    if "Eis" in state.mode:
        f_eis = berechne_eislast(state.A_cross, float(state.d_eis), state.rho_eis)
    else:
        f_eis = 0.0

    f_g_ges = state.F_G + f_eis

    if "Wind" in state.mode:
        load_force = berechne_windeinfluss(f_g_ges, state.F_W)
        winkel = berechne_ausschwingwinkel(f_g_ges, state.F_W)
        w_quer = state.F_W
    else:
        load_force = f_g_ges
        winkel = 0.0
        w_quer = 0.0

    if "Temperatur" in state.mode:
        delta_T = state.T_end - state.T_start
    else:
        delta_T = 0.0

    state.F_Z = round(load_force, 3)
    state.ausschwingwinkel_out = round(winkel, 2)

    H_0_N = state.sigma_0 * state.A_cross
    f_0 = berechne_durchhang_max(F_G=state.F_G, l=state.span_L, H_N=H_0_N)
    state.f_0_out = round(f_0, 3)

    try:
        sigma_1, H_kN = berechne_neue_seilspannung(sigma_0_N_mm2=state.sigma_0, F_G_init_N_m=state.F_G, F_Load_N_m=load_force, E_N_mm2=state.E_modul, L_m=state.span_L, A_mm2=state.A_cross,
            alpha_th=ALPHA_TH, delta_T=delta_T)

        V_kN = (f_g_ges * state.span_L / 2.0) / 1000.0
        W_trans_kN = (w_quer * state.span_L / 2.0) / 1000.0
        F_anker_kN = math.sqrt(H_kN**2 + V_kN**2 + W_trans_kN**2)

        H_1_N = H_kN * 1000.0
        f_1 = berechne_durchhang_max(F_G=load_force, l=state.span_L, H_N=H_1_N)

        state.sigma_1_out = round(sigma_1, 3)
        state.H_zug_out = round(H_kN, 3)
        state.V_kraft_out = round(V_kN, 3)
        state.W_kraft_out = round(W_trans_kN, 3)
        state.F_anker_out = round(F_anker_kN, 3)
        state.f_1_out = round(f_1, 3)

    except Exception as e:
        state.error_msg = f"Fehler bei der Berechnung: {str(e)}"
        state.sigma_1_out = 0.0
        state.H_zug_out = 0.0
        state.V_kraft_out = 0.0
        state.W_kraft_out = 0.0
        state.F_anker_out = 0.0
        state.f_1_out = 0.0

    p_dict = {'F_G':state.F_G, 'F_W':state.F_W, 'T_start':state.T_start, 'sigma_0':state.sigma_0, 'E_modul':state.E_modul, 'span_L':state.span_L, 'A_cross':state.A_cross, 'rho_eis':state.rho_eis,
        'd_eis':state.d_eis}
    state.summary_df = erstelle_uebersichtstabelle(p_dict)


# --- UI BUILDER ---
with tgb.Page() as page:
    tgb.text("# Freileitungs-Seilstatik (IEEE 605)")

    with tgb.part(render="{error_msg != ''}"):
        tgb.text("⚠️ **{error_msg}**")

    tgb.text("Berechnungsmodus")
    tgb.selector(value="{mode}", lov=["Temperatur -20°C", "Temperatur -5°C + Eis", "Wind", "Temperatur +5°C + Wind", "Temperatur 10°C", "Temperatur 80°C", "Temperatur"], on_change=on_change,
        dropdown=True)

    tgb.text("----------------------------------")
    tgb.text("Einwirkungen")
    tgb.text("Gewichtskraft F_G (N/m):")
    tgb.number(value="{F_G}", on_change=on_change, step=0.1)

    with tgb.part(render="{'Eis' in mode}"):
        tgb.text("Eisdichte (kg/m³):")
        tgb.number(value="{rho_eis}", on_change=on_change, step=50.0)
        tgb.text("Eisschichtdicke (mm):")
        tgb.selector(value="{d_eis}", lov=[5, 10, 15, 20], on_change=on_change, dropdown=True)

    with tgb.part(render="{'Wind' in mode}"):
        tgb.text("Windkraft F_W (N/m):")
        tgb.number(value="{F_W}", on_change=on_change, step=0.1)

    with tgb.part(render="{'Temperatur' in mode}"):
        tgb.text("Ausgangstemperatur (°C):")
        tgb.number(value="{T_start}", on_change=on_change, step=5.0)
        tgb.text("Endtemperatur (°C):")
        tgb.number(value="{T_end}", on_change=on_change, step=5.0)

    tgb.text("Aktive Last / F_Z (N/m) [Resultierende Streckenlast]:")
    tgb.number(value="{F_Z}", active=False)

    with tgb.part(render="{'Wind' in mode}"):
        tgb.text("Ausschwingwinkel (°):")
        tgb.number(value="{ausschwingwinkel_out}", active=False)

    tgb.text("----------------------------------")
    tgb.text("Material- & Geometrieparameter")
    tgb.text("Initiale Spannung sigma_0 (N/mm²):")
    tgb.number(value="{sigma_0}", on_change=on_change, step=1.0)

    tgb.text("E-Modul E (N/mm²):")
    tgb.number(value="{E_modul}", on_change=on_change, step=1000.0)

    tgb.text("Seilquerschnitt A (mm²):")
    tgb.number(value="{A_cross}", on_change=on_change, step=10.0)

    tgb.text("Spannweite L (m):")
    tgb.number(value="{span_L}", on_change=on_change, step=5.0)

    tgb.text("Initialer Seildurchhang f_0 (m):")
    tgb.number(value="{f_0_out}", active=False)

    tgb.text("----------------------------------")
    tgb.text("### Einzel-Ergebnisse (Aktiver Modus)")

    tgb.text("Finale Seilspannung sigma_1:")
    tgb.text(value="{sigma_1_out} N/mm²")

    tgb.text("Horizontale Seilzugkraft (H):")
    tgb.text(value="{H_zug_out} kN")

    tgb.text("Vertikale Auflagerkraft (V = F_G_ges * L / 2):")
    tgb.text(value="{V_kraft_out} kN")

    tgb.text("Resultierende Gesamtzugkraft am Abspannpunkt (F_Anker):")
    tgb.text(value="**{F_anker_out} kN**")

    tgb.text("Finaler Seildurchhang f_1:")
    tgb.text(value="**{f_1_out} m**")

    with tgb.part(render="{'Wind' in mode}"):
        tgb.text("Transversale Windauflagerkraft (W = F_W * L / 2):")
        tgb.text(value="{W_kraft_out} kN")

    tgb.text("----------------------------------")
    tgb.text("### Übersicht aller Hauptlastfälle")
    tgb.table(data="{summary_df}", reorderable=False)

# --- APP START ---
if __name__ == "__main__":
    app = Gui(page)
    app.run(title="Freileitungs-Seilstatik", port=5000, host="127.0.0.1", debug=False)