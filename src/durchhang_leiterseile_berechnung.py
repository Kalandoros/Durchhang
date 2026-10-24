from typing import Any
import scipy
import math
import numpy as np
from numpy import ndarray, dtype
import matplotlib.pyplot as plt

# Konstanten
g = scipy.constants.g # 9.81 ist auch möglich

"""
Einheiten:
α: Wärmedehnungszahl in 1/K
α_al: Wärmedehnungszahl von Aluminium in 1/K
α_st: Wärmedehnungszahl von Stahl in 1/K
α_cu: Wärmedehnungszahl von Kupfer in 1/K

A: Querschnitt eines Teilleiters in m^2
A_al: Querschnitt eines Teilleiters aus Aluminium in m^2
A_st: Querschnitt eines Teilleiters aus Stahl in m^2
A_cu: Querschnitt eines Teilleiters aus Kupfer m^2

d: Außendurchmesser von Rohrleitern oder Seildurchmesser in m

ε_th: thermische Seildehnung ε_th (dimensionslos)
ε_el: elastische Seildehnung ε_th (dimensionslos)

E: Elastizitätsmodul in N/m^2
E_al: Elastizitätsmodul von Aluminium N/m^2
E_st: Elastizitätsmodul von Stahl N/m^2
E_cu: Elastizitätsmodul von Kupfer N/m^2

F_s: statische Seilzugkraft in einem Hauptleiter in N
F_h: horizontale Komponente der Seilzugkraft in einem Hauptleiter in N
F_v: vertikale Komponente der Seilzugkraft in einem Hauptleiter in N

f: Lotrechter Seildurchhang in m
f_s: Seildurchhang am Scheitel in m
f_x: Seildurchhang an der Stelle x in m
f_min: Minimaler Seildurchhang in m
f_max: Minimaler Seildurchhang in m

g: Normfallbeschleunigung in m/s^2

L_k: Länge einer Isolatorkette in m
L_kx: Länge einer Isolatorkette projiziert in die horizontale in m

l: Abstand der Abspannstellen in m

m_s: Massenbelag eines Teilleiters in kg/m
m_si: Massenbelag eines Teilleiters mit Eisansatz in kg/m

n: Anzahl der Teilleiter eines Hauptleiters (dimensionslos)

ρ: Dichte eines Teilleiters in kg/(m * mm^2)
ρ_al: Dichte eines Teilleiters aus Aluminium kg/(m * mm^2)
ρ_st: Dichte eines Teilleiters aus Stahl kg/(m * mm^2)
ρ_cu: Dichte eines Teilleiters aus Kupfer kg/(m * mm^2)

q: Querschnittsverhältnis von Leitermaterialien (dimensionslos)

σ: Seilzugspannung eines Teilleiters in N/mm^2
σ_hoehst: Zulässige Höchstseilzugspannung Seilzugspannung eines Teilleiters in N/mm^2
σ_mittel: Zulässige Mittelseilzugspannung eines Teilleiters in N/mm^2
σ_dauer: Zulässige Dauerseilzugspannung eines Teilleiters in N/mm^2

ϑ: Temperatur in °C
ϑ_niedrig: Temperatur in °C
ϑ_hoch: Temperatur in °C






f_es: statischer Ersatz-Seildurchhang in Spannfeldmitte in m

l: Mittenabstand der Stützpunkte in m
σ_fin: niedrigste Spannung, ab der der Elastizitätsmodul konstant wird, in N/m^2
N: Steifigkeitsnorm einer Anordnung mit Leiterseilen in 1/N


l_v: Seil(bogen)länge der Schlaufe in m
h: Schlaufenhöhe in m
w: Schlaufenbreite in m

h: Horizontale Seilkraft in N
w: Leitergewicht pro Laengeneinheit in N/m
x: Position in m
"""


def kettenleiter(h: float|int, w: float|int, x: float|int) -> float|int:
    """
    Berechnet die Position y(x) einer Kettenlinie.
    h: Horizontale Seilkraft in N
    w: Leitergewicht pro Laengeneinheit in N/m
    x: Position in m
    """
    y = h / w * (math.cosh((w * x) / h) - 1)
    return y


def seillaengenaenderung_temperatureinfluss(s: float, t: float) -> float:
    """
    Laengenaenderung aufgrund des Temperatureinflusses.
    s: Leiterseillaenge bei Referenztemperatur in m
    t: Temperatur in degC
    """
    t_ref: float = 10
    alpha_a: float = 23e-6  # für Aluminium
    delta_s = alpha_a * (t - t_ref) * s
    print("Laengenaenderung aufgrund des Temperatureinflusses in m:", delta_s)
    return delta_s


def kettenleiter_gerade_gesamt(h: float, w: float, s: float) -> tuple[ndarray[Any, dtype[Any]], list[float]]:
    """
    Berechnet den gesamten geraden Kettenleiter.
    h: Horizontale Seilkraft in N
    w: Leitergewicht pro Laengeneinheit in N/m
    s: Spannweite in m
    """
    if h <= 0 or w <= 0 or s <= 0:
        raise ValueError("h, w und s muessen groesser als 0 sein.")

    x = np.arange(-s / 2, s / 2, 0.001)
    y = [kettenleiter(h, w, item) for item in x]
    kettenleiter_plot(x, y)
    return x, y


def kettenleiter_schraeg_gesamt(h: float, w: float, s: float, h_diff: float, art: str) \
        -> tuple[ndarray[Any, dtype[Any]], list[float]]:
    """
    Berechnet den gesamten schraegen Kettenleiter.
    h: Horizontale Seilkraft in N
    w: Leitergewicht pro Laengeneinheit in N/m
    s: Spannweite in m
    h_diff: Hoehendifferenz der Aufhaengepunkte in m (Betrag)
    art: "links hoch" oder "rechts hoch"
    """
    if h <= 0 or w <= 0 or s <= 0:
        raise ValueError("h, w und s muessen groesser als 0 sein.")
    if h_diff < 0:
        raise ValueError("h_diff muss groesser oder gleich 0 sein.")

    if art == "links hoch":
        # Linker Aufhaengepunkt ist hoeher, daher delta = y(rechts) - y(links) < 0.
        delta = -h_diff
    elif art == "rechts hoch":
        # Rechter Aufhaengepunkt ist hoeher, daher delta = y(rechts) - y(links) > 0.
        delta = h_diff
    else:
        raise ValueError("Art der Schraeglage angeben: 'links hoch' oder 'rechts hoch'.")

    # Seitlicher Versatz der Kettenlinie relativ zur Spannweitenmitte.
    shift = (h / w) * math.asinh((delta * w) / (2 * h * math.sinh((w * s) / (2 * h))))
    print("Seitlicher Versatz in m:", shift)

    # Spannweite bleibt immer gleich; der Versatz geht in das Kettenlinien-Argument ein.
    x = np.arange(-s / 2, s / 2, 0.001)
    y = [kettenleiter(h, w, item + shift) for item in x]
    kettenleiter_plot(x, y)
    return x, y


def durchhang(h: float, w: float, s: float) -> float:
    """
    Berechnet den maximalen Durchhang.
    h: Horizontale Seilkraft in N
    w: Leitergewicht pro Laengeneinheit in N/m
    s: Spannweite in m
    """
    if h <= 0 or w <= 0 or s <= 0:
        raise ValueError("h, w und s muessen groesser als 0 sein.")

    d = h / w * (math.cosh((w * s) / (2 * h)) - 1)
    print("Maximaler Durchhang in m:", d)
    return d


def leiterlaenge(h: float, w: float, s: float) -> float:
    """
    Berechnet die Laenge des Leiterseils.
    h: Horizontale Seilkraft in N
    w: Leitergewicht pro Laengeneinheit in N/m
    s: Spannweite in m
    """
    if h <= 0 or w <= 0 or s <= 0:
        raise ValueError("h, w und s muessen groesser als 0 sein.")

    l = ((2 * h) / w) * math.sinh((w * s) / (2 * h))
    return l


def leiterlaenge_spannweite_differenz(h: float, w: float, s: float) -> float:
    """
    Differenz zwischen Leiterseillaenge und Spannweite.
    """
    if h <= 0 or w <= 0 or s <= 0:
        raise ValueError("h, w und s muessen groesser als 0 sein.")

    l_s = (((2 * h) / w) * math.sinh((w * s) / (2 * h))) - s
    l_s_pct = l_s / s * 100
    print("Differenz Leiterseillaenge und Spannweite in m:", l_s)
    print("Differenz Leiterseillaenge und Spannweite in %:", l_s_pct)
    return l_s


def seilzugspannung_gesamt_approx(h: float, w: float, d: float) -> float:
    """
    Approximation der gesamten Seilzugspannung.
    """
    t = h + (w * d)
    print("Gesamte Zugspannung in N:", t)
    return t


def seilzugspannung_gesamt_exact(h: float, w: float, s: float) -> float:
    """
    Exakte gesamte Seilzugspannung.
    """
    if h <= 0 or w <= 0 or s <= 0:
        raise ValueError("h, w und s muessen groesser als 0 sein.")

    t = h + w * ((h / w) * math.cosh((w * s) / (2 * h)) - (h / w))
    print("Gesamte Zugspannung in N:", t)
    return t


def kettenleiter_plot(x, y):
    if len(x) == 0 or len(y) == 0:
        raise ValueError("x und y duerfen nicht leer sein.")
    if len(x) != len(y):
        raise ValueError("x und y muessen gleich lang sein.")

    y_max = max(y)
    fig = plt.figure(figsize=(8, 8))
    fig.canvas.header_visible = False
    plt.title("Diagramm Leiterseil und Durchhang")
    fig.canvas.toolbar_position = "top"
    plt.grid(which="major", axis="both")
    plt.plot(x, y, label="Leiterseil")
    plt.scatter(0, y_max, s=15, color="red")
    plt.scatter(0, 0, s=15, color="red")
    plt.vlines(x=0, ymin=0, ymax=y_max, color="black", linestyle="dashed", linewidth=1, label="Durchhang")
    h1, l1 = plt.gca().get_legend_handles_labels()
    plt.xlabel("Distanz in m")
    plt.xticks(rotation=65)

    if y_max > 0:
        plt.ylim(-y_max * 0.25, y_max * 1.25)

    plt.ylabel("Hoehe in m")
    plt.legend(h1, l1)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show()


if __name__ == "__main__":
    # Beispiele:
    # kettenleiter_gerade_gesamt(h=28000, w=15.97, s=300)
    # durchhang(h=28000, w=15.97, s=300)
    # leiterlaenge(h=28000, w=15.97, s=300)
    # leiterlaenge_spannweite_differenz(h=28000, w=15.97, s=300)
    # seilzugspannung_gesamt_approx(h=28000, w=15.97, d=6.42)
    # seilzugspannung_gesamt_exact(h=28000, w=15.97, s=300)

    # kettenleiter_schraeg_gesamt(h=28000, w=15.97, s=300, h_diff=10, art="links hoch")
    # kettenleiter_schraeg_gesamt(h=28000, w=15.97, s=300, h_diff=10, art="rechts hoch")

    # Test fuer Birr:
    kettenleiter_schraeg_gesamt(h=1000, w=21.8286, s=16.6, h_diff=3.5, art="links hoch")
