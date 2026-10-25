from taipy.gui import Gui, State
import taipy.gui.builder as tgb
import math
import types
import pandas as pd
import json
from pathlib import Path
from scipy.optimize import root_scalar

# File to store current State for external access (e.g., Jupyter notebook)
_STATE_FILE = Path(__file__).parent / "state.json"


# --- LOGIK & BERECHNUNG ---

def seilzugspannung_gesamt_exact(h: float|int, w: float|int, s: float|int) -> float|int:
    """Exakte Gesamtzugkraft T an den Aufhängungspunkten in N."""
    if h <= 0 or w <= 0 or s <= 0:
        return 0.0
    return h * math.cosh((w * s) / (2.0 * h))


def ungedehnte_seillänge_exact(H: float|int, w: float|int, s: float|int, E_Pa: float|int, A_m2: float|int) -> float|int:
    """Berechnet die ungedehnte Seillänge der elastischen Kettenlinie."""
    if H <= 0 or w <= 0 or s <= 0:
        return s
    S = (2.0 * H / w) * math.sinh((w * s) / (2.0 * H))
    delta_L_el = (H / (2.0 * E_Pa * A_m2)) * (s + (H / w) * math.sinh((w * s) / H))
    return S - delta_L_el


def berechne_eislast(A_mm2: float|int, d_eis_mm: float|int, rho_eis: float|int) -> float|int:
    """Berechnet die Eislast F_Eis in N/m für EINEN Teilleiter."""
    if d_eis_mm <= 0 or A_mm2 <= 0 or rho_eis <= 0:
        return 0.0

    d_seil_mm = math.sqrt((4.0 * A_mm2) / math.pi)
    A_eis_mm2 = math.pi * d_eis_mm * (d_seil_mm + d_eis_mm)
    A_eis_m2 = A_eis_mm2 * 1e-6

    F_eis_N_m = A_eis_m2 * rho_eis * 9.81
    return F_eis_N_m


def berechne_windeinfluss(F_G: float|int, F_W: float|int) -> float|int:
    """Resultierende Last aus Eigengewicht und Windlast nach Pythagoras."""
    return math.sqrt(F_G**2 + F_W**2)


def berechne_ausschwingwinkel(F_G: float|int, F_W: float|int) -> float|int:
    """Ausschwingwinkel des Leiters durch Wind in Grad."""
    if F_W <= 0:
        return 90.0
    return math.degrees(math.atan(F_G / F_W))


def berechne_durchhang_max(method: str, F_G: float|int, l: float|int, H_N: float|int) -> float|int:
    """Berechnet den maximalen Durchhang f1 (Parabel oder exakte Kettenlinie)."""
    if H_N <= 0 or F_G <= 0 or l <= 0:
        return 0.0
    if method == "Parabel - Näherung":
        return ((l**2 * F_G) / (8.0 * H_N)) + ((l**4 * F_G**3) / (384.0 * H_N**3))
    else:
        return (H_N / F_G) * (math.cosh((F_G * l) / (2.0 * H_N)) - 1.0)


def berechne_neue_seilspannung_parabel(sigma_0_N_mm2: float|int, F_G_init_N_m: float|int, F_Load_N_m: float|int, E_N_mm2: float|int, L_m: float|int, A_mm2: float|int, alpha_th: float|int, delta_T: float|int) -> tuple[float|int, float|int]:
    """Löst die kubische Zustandsgleichung (Parabelnäherung nach Ernst)."""
    sigma_0 = sigma_0_N_mm2 * 1e6  # Pa
    E = E_N_mm2 * 1e6  # Pa
    A = A_mm2 * 1e-6  # m²
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


def berechne_neue_seilspannung_kettenlinie(sigma_0_N_mm2: float, F_G_init_N_m: float, F_Load_N_m: float, E_N_mm2: float, L_m: float, A_mm2: float, alpha_th: float, delta_T: float) -> tuple[
    float, float]:
    """Löst die exakte Kettenlinien-Zustandsgleichung via SciPy."""
    sigma_0 = sigma_0_N_mm2 * 1e6
    E_Pa = E_N_mm2 * 1e6
    A_m2 = A_mm2 * 1e-6
    H_0 = sigma_0 * A_m2

    w_0 = F_G_init_N_m
    w_1 = F_Load_N_m

    L0_un = ungedehnte_seillänge_exact(H_0, w_0, L_m, E_Pa, A_m2)
    L1_un_soll = L0_un * (1.0 + alpha_th * delta_T)

    sigma_1_parabel, _ = berechne_neue_seilspannung_parabel(sigma_0_N_mm2, F_G_init_N_m, F_Load_N_m, E_N_mm2, L_m, A_mm2, alpha_th, delta_T)
    H1_init = sigma_1_parabel * 1e6 * A_m2
    if H1_init <= 0:
        H1_init = 100.0

    def zielfunktion_kettenlinie(H1):
        if H1 <= 0:
            return 1e9
        L1_un_ist = ungedehnte_seillänge_exact(H1, w_1, L_m, E_Pa, A_m2)
        return L1_un_ist - L1_un_soll

    sol = root_scalar(zielfunktion_kettenlinie, x0=H1_init, x1=H1_init * 1.0001, method='secant', xtol=1e-6)

    H_1_exact = sol.root
    sigma_1_N_mm2 = (H_1_exact / A_m2) / 1e6
    H_zug_kN = H_1_exact / 1000.0

    return sigma_1_N_mm2, H_zug_kN


def berechne_neue_seilspannung(method: str, sigma_0_N_mm2: float|int, F_G_init_N_m: float|int, F_Load_N_m: float|int, E_N_mm2: float|int, L_m: float|int, A_mm2: float|int, alpha_th: float|int, delta_T: float|int) -> tuple[
    float|int, float|int]:
    """Wählt je nach Methode das Berechnungsverfahren aus."""
    if method == "Parabel - Näherung":
        return berechne_neue_seilspannung_parabel(sigma_0_N_mm2, F_G_init_N_m, F_Load_N_m, E_N_mm2, L_m, A_mm2, alpha_th, delta_T)
    else:
        return berechne_neue_seilspannung_kettenlinie(sigma_0_N_mm2, F_G_init_N_m, F_Load_N_m, E_N_mm2, L_m, A_mm2, alpha_th, delta_T)


def berechne_kraefte(method: str, f_g_ges: float|int, w_quer: float|int, load_force: float|int, L: float|int, H_kN: float|int) -> tuple[float|int, float|int, float|int]:
    """Berechnet V_kN, W_trans_kN und F_anker_kN basierend auf der gewählten Methode."""
    H_N = H_kN * 1000.0

    if method == "Parabel - Näherung":
        V_kN = (f_g_ges * L / 2.0) / 1000.0
        W_trans_kN = (w_quer * L / 2.0) / 1000.0
        F_anker_kN = math.sqrt(H_kN**2 + V_kN**2 + W_trans_kN**2)
    else:
        if H_N > 0 and f_g_ges > 0:
            V_kN = (H_N * math.sinh((f_g_ges * L) / (2.0 * H_N))) / 1000.0
        else:
            V_kN = 0.0

        W_trans_kN = (w_quer * L / 2.0) / 1000.0

        if H_N > 0 and load_force > 0:
            T_kN = (H_N * math.cosh((load_force * L) / (2.0 * H_N))) / 1000.0
        else:
            T_kN = H_kN

        if w_quer > 0:
            F_anker_kN = math.sqrt(T_kN**2 + W_trans_kN**2)
        else:
            F_anker_kN = T_kN

    return V_kN, W_trans_kN, F_anker_kN


ALPHA_TH = 0.000023


def berechne_einzelnen_modus(m_name: str, p: dict) -> dict:
    method = p.get('calc_method', "Kettenlinie - Exakt")
    n_sub = int(p.get('n_sub', 1))
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

    # Lasten pro Teilleiter
    f_eis_1 = berechne_eislast(p['A_cross'], float(p['d_eis']), p['rho_eis']) if has_eis else 0.0
    f_g_1 = p['F_G'] + f_eis_1
    w_quer_1 = p['F_W'] if has_wind else 0.0
    load_force_1 = berechne_windeinfluss(f_g_1, p['F_W']) if has_wind else f_g_1
    delta_T = T_end - p['T_start'] if "Temperatur" in m_name else 0.0

    # Bündel-Gesamtlasten
    f_g_ges_buendel = n_sub * f_g_1
    w_quer_buendel = n_sub * w_quer_1
    load_force_buendel = n_sub * load_force_1
    A_cross_buendel = n_sub * p['A_cross']
    F_G_init_buendel = n_sub * p['F_G']

    try:
        # Seilspannung σ₁ [N/mm²] (identisch für Teilleiter und Bündel)
        sigma_1, H_kN_buendel = berechne_neue_seilspannung(method=method, sigma_0_N_mm2=p['sigma_0'], F_G_init_N_m=F_G_init_buendel, F_Load_N_m=load_force_buendel, E_N_mm2=p['E_modul'],
            L_m=p['span_L'], A_mm2=A_cross_buendel, alpha_th=ALPHA_TH, delta_T=delta_T)

        # Gesamtkräfte am Bündel
        V_kN_buendel, W_trans_kN_buendel, F_anker_kN_buendel = berechne_kraefte(method=method, f_g_ges=f_g_ges_buendel, w_quer=w_quer_buendel, load_force=load_force_buendel, L=p['span_L'],
            H_kN=H_kN_buendel)

        H_1_N_buendel = H_kN_buendel * 1000.0
        f_1 = berechne_durchhang_max(method=method, F_G=load_force_buendel, l=p['span_L'], H_N=H_1_N_buendel)

        return {"sigma_1":round(sigma_1, 3), "H_zug":round(H_kN_buendel, 3), "V_kraft":round(V_kN_buendel, 3), "W_kraft":round(W_trans_kN_buendel, 3), "F_anker":round(F_anker_kN_buendel, 3),
            "f_1":round(f_1, 3)}
    except Exception:
        return {"sigma_1":0.0, "H_zug":0.0, "V_kraft":0.0, "W_kraft":0.0, "F_anker":0.0, "f_1":0.0}


def erstelle_uebersichtstabelle(p: dict) -> pd.DataFrame:
    spalten_modi = ["Temperatur -20°C", "Temperatur -5°C + Eis", "Temperatur +5°C + Wind", "Temperatur 10°C", "Temperatur 80°C"]

    results = {m:berechne_einzelnen_modus(m, p) for m in spalten_modi}

    zeilen_definition = [("Finale Seilspannung σ₁ [N/mm²]", "sigma_1"), ("Gesamte Horizontalkraft H_Bündel [kN]", "H_zug"), ("Gesamte Vertikalkraft V_Bündel [kN]", "V_kraft"),
        ("Gesamte Ankerkraft F_Anker,Bündel [kN]", "F_anker"), ("Finaler Seildurchhang f₁ [m]", "f_1"), ("Gesamte Windkraft W_Bündel [kN]", "W_kraft")]

    data = {"Kenngröße / Parameter":[z[0] for z in zeilen_definition]}
    for m in spalten_modi:
        data[m] = [results[m][key] for _, key in zeilen_definition]

    return pd.DataFrame(data)


# --- STATE INITIALISIERUNG ---
calc_method = "Kettenlinie - Exakt"
mode = "Wind"
n_sub = 1  # Anzahl Teilleiter
F_G = 10.849  # N/m pro Teilleiter
F_W = 24.544  # N/m pro Teilleiter
F_Z = 26.920
T_start = 10.0
T_end = -20.0
sigma_0 = 12.5
E_modul = 55000.0
span_L = 20.0
A_cross = 400.0  # mm² pro Teilleiter

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

init_params = {'calc_method':calc_method, 'n_sub':n_sub, 'F_G':F_G, 'F_W':F_W, 'T_start':T_start, 'sigma_0':sigma_0, 'E_modul':E_modul, 'span_L':span_L, 'A_cross':A_cross, 'rho_eis':rho_eis,
    'd_eis':d_eis}
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

    n_sub_val = int(state.n_sub)

    if "Eis" in state.mode:
        f_eis_1 = berechne_eislast(state.A_cross, float(state.d_eis), state.rho_eis)
    else:
        f_eis_1 = 0.0

    f_g_1 = state.F_G + f_eis_1

    if "Wind" in state.mode:
        load_force_1 = berechne_windeinfluss(f_g_1, state.F_W)
        winkel = berechne_ausschwingwinkel(f_g_1, state.F_W)
        w_quer_1 = state.F_W
    else:
        load_force_1 = f_g_1
        winkel = 0.0
        w_quer_1 = 0.0

    if "Temperatur" in state.mode:
        delta_T = state.T_end - state.T_start
    else:
        delta_T = 0.0

    state.F_Z = round(load_force_1 * n_sub_val, 3)
    state.ausschwingwinkel_out = round(winkel, 2)

    # Bündel-Gesamtwerte
    A_cross_buendel = n_sub_val * state.A_cross
    F_G_init_buendel = n_sub_val * state.F_G
    f_g_ges_buendel = n_sub_val * f_g_1
    w_quer_buendel = n_sub_val * w_quer_1
    load_force_buendel = n_sub_val * load_force_1

    H_0_N_buendel = state.sigma_0 * A_cross_buendel
    f_0 = berechne_durchhang_max(method=state.calc_method, F_G=n_sub_val * state.F_G, l=state.span_L, H_N=H_0_N_buendel)
    state.f_0_out = round(f_0, 3)

    try:
        sigma_1, H_kN_buendel = berechne_neue_seilspannung(method=state.calc_method, sigma_0_N_mm2=state.sigma_0, F_G_init_N_m=F_G_init_buendel, F_Load_N_m=load_force_buendel, E_N_mm2=state.E_modul,
            L_m=state.span_L, A_mm2=A_cross_buendel, alpha_th=ALPHA_TH, delta_T=delta_T)

        V_kN_buendel, W_trans_kN_buendel, F_anker_kN_buendel = berechne_kraefte(method=state.calc_method, f_g_ges=f_g_ges_buendel, w_quer=w_quer_buendel, load_force=load_force_buendel, L=state.span_L,
            H_kN=H_kN_buendel)

        H_1_N_buendel = H_kN_buendel * 1000.0
        f_1 = berechne_durchhang_max(method=state.calc_method, F_G=load_force_buendel, l=state.span_L, H_N=H_1_N_buendel)

        state.sigma_1_out = round(sigma_1, 3)
        state.H_zug_out = round(H_kN_buendel, 3)
        state.V_kraft_out = round(V_kN_buendel, 3)
        state.W_kraft_out = round(W_trans_kN_buendel, 3)
        state.F_anker_out = round(F_anker_kN_buendel, 3)
        state.f_1_out = round(f_1, 3)

    except Exception as e:
        state.error_msg = f"Fehler bei der Berechnung: {str(e)}"
        state.sigma_1_out = 0.0
        state.H_zug_out = 0.0
        state.V_kraft_out = 0.0
        state.W_kraft_out = 0.0
        state.F_anker_out = 0.0
        state.f_1_out = 0.0

    p_dict = {'calc_method':state.calc_method, 'n_sub':n_sub_val, 'F_G':state.F_G, 'F_W':state.F_W, 'T_start':state.T_start, 'sigma_0':state.sigma_0, 'E_modul':state.E_modul, 'span_L':state.span_L,
        'A_cross':state.A_cross, 'rho_eis':state.rho_eis, 'd_eis':state.d_eis}
    state.summary_df = erstelle_uebersichtstabelle(p_dict)

    # Write current State to JSON file for external access (e.g., Jupyter notebook)
    # Only include JSON-serializable user variables (numbers, strings, booleans, lists, dicts)
    _JSON_TYPES = (int, float, str, bool, type(None))
    state_dict = {}
    user_vars = object.__getattribute__(state, '_user_var_list')
    for key in user_vars:
        if key == '__name__':
            continue
        val = getattr(state, key, None)
        if not isinstance(val, _JSON_TYPES):
            continue  # Skip anything that is not a basic JSON type
        state_dict[key] = val
    with open(_STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state_dict, f, ensure_ascii=False, indent=2)


# --- UI BUILDER ---
with tgb.Page() as page:
    tgb.text("# Freileitungs-Seilstatik (IEEE 605)")

    with tgb.part(render="{error_msg != ''}"):
        tgb.text("⚠️ **{error_msg}**")

    tgb.text("Berechnungsmethode")
    tgb.selector(value="{calc_method}", lov=["Kettenlinie - Exakt", "Parabel - Näherung"], on_change=on_change, dropdown=True)

    tgb.text("Berechnungsmodus")
    tgb.selector(value="{mode}", lov=["Temperatur -20°C", "Temperatur -5°C + Eis", "Wind", "Temperatur +5°C + Wind", "Temperatur 10°C", "Temperatur 80°C", "Temperatur"], on_change=on_change,
        dropdown=True)

    tgb.text("----------------------------------")
    tgb.text("Einwirkungen (pro Teilleiter)")
    tgb.text("Gewichtskraft F_G (N/m pro Teilleiter):")
    tgb.number(value="{F_G}", on_change=on_change, step=0.1)

    with tgb.part(render="{'Eis' in mode}"):
        tgb.text("Eisdichte (kg/m³):")
        tgb.number(value="{rho_eis}", on_change=on_change, step=50.0)
        tgb.text("Eisschichtdicke (mm):")
        tgb.selector(value="{d_eis}", lov=[5, 10, 15, 20], on_change=on_change, dropdown=True)

    with tgb.part(render="{'Wind' in mode}"):
        tgb.text("Windkraft F_W (N/m pro Teilleiter):")
        tgb.number(value="{F_W}", on_change=on_change, step=0.1)

    with tgb.part(render="{'Temperatur' in mode}"):
        tgb.text("Ausgangstemperatur (°C):")
        tgb.number(value="{T_start}", on_change=on_change, step=5.0)
        tgb.text("Endtemperatur (°C):")
        tgb.number(value="{T_end}", on_change=on_change, step=5.0)

    tgb.text("Aktive Last / F_Z (N/m) [Gesamte Streckenlast des Bündels]:")
    tgb.number(value="{F_Z}", active=False)

    with tgb.part(render="{'Wind' in mode}"):
        tgb.text("Ausschwingwinkel (°):")
        tgb.number(value="{ausschwingwinkel_out}", active=False)

    tgb.text("----------------------------------")
    tgb.text("Material- & Geometrieparameter")

    tgb.text("Anzahl Teilleiter n:")
    tgb.selector(value="{n_sub}", lov=[1, 2, 3, 4, 6, 8], on_change=on_change, dropdown=True)

    tgb.text("Initiale Spannung sigma_0 (N/mm²):")
    tgb.number(value="{sigma_0}", on_change=on_change, step=1.0)

    tgb.text("E-Modul E (N/mm²):")
    tgb.number(value="{E_modul}", on_change=on_change, step=1000.0)

    tgb.text("Teilleiter-Querschnitt A1 (mm² pro Teilleiter):")
    tgb.number(value="{A_cross}", on_change=on_change, step=10.0)

    tgb.text("Spannweite L (m):")
    tgb.number(value="{span_L}", on_change=on_change, step=5.0)

    tgb.text("Initialer Seildurchhang f_0 (m):")
    tgb.number(value="{f_0_out}", active=False)

    tgb.text("----------------------------------")
    tgb.text("### Einzel-Ergebnisse (Aktiver Modus)")

    tgb.text("Finale Seilspannung sigma_1:")
    tgb.text(value="{sigma_1_out} N/mm²")

    tgb.text("Gesamte Horizontale Seilzugkraft (H_Bündel):")
    tgb.text(value="{H_zug_out} kN")

    tgb.text("Gesamte Vertikale Auflagerkraft (V_Bündel):")
    tgb.text(value="{V_kraft_out} kN")

    tgb.text("Resultierende Gesamtzugkraft am Abspannpunkt (F_Anker,Bündel):")
    tgb.text(value="**{F_anker_out} kN**")

    tgb.text("Finaler Seildurchhang f_1:")
    tgb.text(value="**{f_1_out} m**")

    with tgb.part(render="{'Wind' in mode}"):
        tgb.text("Gesamte Transversale Windauflagerkraft (W_Bündel):")
        tgb.text(value="{W_kraft_out} kN")

    tgb.text("----------------------------------")
    tgb.text("### Übersicht aller Hauptlastfälle (Bündel-Gesamtkräfte)")
    tgb.table(data="{summary_df}", reorderable=False)

# --- APP START ---
if __name__ == "__main__":
    app = Gui(page)
    app.run(title="Freileitungs-Seilstatik", port=5000, host="127.0.0.1", debug=False)