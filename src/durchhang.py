from typing import Any

import math
import numpy as np
from numpy import ndarray, dtype
import matplotlib.pyplot as plt


def kettenleiter(h: float, w: float, x: float) -> float:
    """
    Funktion zur Berechnung der Postion y(x) eines geraden Kettenleiters
    h: Horizontale Seilkraft in N
    w: Leitergewicht pro Längeneinheit in N/m
    x: Postion x für die y bestimmt werden soll in m
    Erläuterung zu h: Gemäss Cigre 324 WG B2-12 Kapitel 4.1 kann die horizontale Kraft im unbelasteten Zustand
    mit 20% der rechnerischen Bruchkraft bei 15°C angenommen werden.
    """
    y = h / w * ((math.cosh((w * x) / h)) - 1)
    return y


def temperatureinfluss(w: float, t: float) -> float:
    reference_temperatur: float = 10
    alpha: float = 23E-6                                # für Aluminium
    delta_s = alpha * (reference_temperatur - t) * w


def kettenleiter_gerade_gesamt(h: float, w: float, s: float) -> tuple[ndarray[Any, dtype[Any]], list[float]]:
    """
    Funktion zur Berechnung des gesamten geraden Kettenleiters
    h: Horizontale Seilkraft in N
    w: Leitergewicht pro Längeneinheit in N/m
    s: Spannweite in m
    Erläuterung zu h: Gemäss Cigre 324 WG B2-12 Kapitel 4.1 kann die horizontale Kraft im unbelasteten Zustand
    mit 20% der rechnerischen Bruchkraft bei 15°C angenommen werden.
    """
    x = np.arange(-s / 2, s / 2, 0.001)
    y = []
    for count, item in enumerate(x):
        y_yield = kettenleiter(h, w, item)
        y.append(y_yield)
    kettenleiter_plot(x, y)
    return x, y


def kettenleiter_schraeg_gesamt(h: float, w: float, s: float, h_diff: float, art: str) \
        -> tuple[ndarray[Any, dtype[Any]], list[float]]:
    """
    Funktion zur Berechnung des gesamten schrägen Kettenleiters
    h: Horizontale Seilkraft in N
    w: Leitergewicht pro Längeneinheit in N/m
    s: Spannweite in m
    Erläuterung zu h: Gemäss Cigre 324 WG B2-12 Kapitel 4.1 kann die horizontale Kraft im unbelasteten Zustand
    mit 20% der rechnerischen Bruchkraft bei 15°C angenommen werden.
    """
    # TODO: Schräger Durchhang für rechte höhere Seite einfügen
    # TODO: Durchhang für Schräge Kettenleiter hinzufügen
    if art == "links hoch":
        xr = (s / 2) - ((h / w) * (math.asinh((h_diff / 2) / ((h / w) * math.sinh((s / 2) / (h / w))))))
        print("xr in m:", xr)
        x = np.arange(-s - xr / 2, s / 2, 0.001)
    elif art == "rechts hoch":
        xl = (s / 2) - ((h / w) * (math.asinh((h_diff / 2) / ((h / w) * math.sinh((s / 2) / (h / w))))))
        print("xl in m:", xl)
        x = np.arange(-s / 2, s - xl / 2, 0.001)
    else:
        print("Art der Schräge angeben (links hoch, rechts hoch)!")
    y = []
    for count, item in enumerate(x):
        y_yield = kettenleiter(h, w, item)
        y.append(y_yield)
    kettenleiter_plot(x, y)
    return x, y


def durchhang(h: float, w: float, s: float) -> float:
    """
    Funktion zur Berechnung des maximalen Durchhangs
    h: Horizontale Seilkraft in N
    w: Leitergewicht pro Längeneinheit in N/m
    s: Spannweite in m
    Erläuterung zu h: Gemäss Cigre 324 WG B2-12 Kapitel 4.1 kann die horizontale Kraft im unbelasteten Zustand
    mit 20% der rechnerischen Bruchkraft bei 15°C angenommen werden.
    """
    d = h / w * ((math.cosh((w * s) / (2 * h))) - 1)
    print("Maximaler Durchhang in m:", d)
    return d


def leiterlaenge(h: float, w: float, s: float) -> float:
    """
    Funktion zur Berechnung der Länge des Leiterseils
    h: Horizontale Seilkraft in N
    w: Leitergewicht pro Längeneinheit in N/m
    s: Spannweite in m
    Erläuterung zu h: Gemäss Cigre 324 WG B2-12 Kapitel 4.1 kann die horizontale Kraft im unbelasteten Zustand
    mit 20% der rechnerischen Bruchkraft bei 15°C angenommen werden.
    """
    l = ((2 * h) / w) * (math.sinh((w * s) / (2 * h)))
    print("Länge des Leiterseiles in m:", l)
    return l


def leiterlaenge_spannweite_differenz(h: float, w: float, s: float) -> float:
    """
        Funktion zur Berechnung der Differenz zwischen Länge des Leiterseils und der Spannweite
        h: Horizontale Seilkraft in N
        w: Leitergewicht pro Längeneinheit in N/m
        s: Spannweite in m
        Erläuterung zu h: Gemäss Cigre 324 WG B2-12 Kapitel 4.1 kann die horizontale Kraft im unbelasteten Zustand
        mit 20% der rechnerischen Bruchkraft bei 15°C angenommen werden.
    """
    l_s = (((2 * h) / w) * (math.sinh((w * s) / (2 * h)))) - s
    l_s_pct = l_s / s * 100
    print("Differenz Leiterseillänge und Spannweite in m:", l_s)
    print("Differenz Leiterseillänge und Spannweite in %:", l_s_pct)
    return l_s


def seilzugspannung_gesamt_approx(h: float, w: float, d: float) -> float:
    """
        Funktion zur Berechnung der gesamten Seilzugspannung (horizontale und vertikale Komponente)
        h: Horizontale Kraft in N/mm2
        w: Leitergewicht pro Längeneinheit in N/m
        d: Durchhang in m
        Erläuterung zu h: Gemäss Cigre 324 WG B2-12 Kapitel 4.1 kann die horizontale Kraft im unbelasteten Zustand
        mit 20% der rechnerischen Bruchkraft bei 15°C angenommen werden.
    """
    t = h + (w * d)
    print("Gesamten Zugspannung in N:", t)
    return t


def seilzugspannung_gesamt_exact(h: float, w: float, s: float) -> float:
    """
        Funktion zur Berechnung der gesamten Seilzugspannung (horizontale und vertikale Komponente)
        h: Horizontale Kraft in N/mm2
        w: Leitergewicht pro Längeneinheit in N/m
        s: Spannweite in m
        Erläuterung zu h: Gemäss Cigre 324 WG B2-12 Kapitel 4.1 kann die horizontale Kraft im unbelasteten Zustand
        mit 20% der rechnerischen Bruchkraft bei 15°C angenommen werden.
    """
    t = h + w * ((h / w) * (math.cosh((w * s) / (2 * h))) - (h / w))
    print("Gesamten Zugspannung in N:", t)
    return t


def kettenleiter_plot(x, y):
    fig = plt.figure(figsize=(8, 8))
    fig.canvas.header_visible = False
    plt.title('Diagramm Leiterseil und Durchhang')
    fig.canvas.toolbar_position = 'top'
    plt.grid(which='major', axis='both')
    plt.plot(x, y, label="Leiterseil")
    plt.scatter(0, max(y), s=15, color='red')
    plt.scatter(0, 0, s=15, color='red')
    plt.vlines(x=0, ymin=0, ymax=max(y), color='black', linestyle='dashed', linewidth=1, label='Durchhang')
    h1, l1 = plt.gca().get_legend_handles_labels()
    plt.xlabel("Distanz in m")
    plt.xticks(rotation=65)
    plt.ylim(-max(y) * 0.5 / 2, max(y) * 2.5 / 2)
    plt.ylabel("Höhe in m")
    plt.legend(h1, l1)
    # plt.gca().set_aspect('equal')
    # plt.subplots_adjust(bottom=.25, left=.25)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show()


if __name__ == "__main__":
    # Beispielrechnung gemäss Cigre 324 WG B2-12 Calculation of Sag-tension for Overhead Power Lines - Seite 10 →
    # Verifiziert
    kettenleiter_gerade_gesamt(h=28000, w=15.97, s=300)
    # Beispielrechnung gemäss Cigre 324 WG B2-12 Calculation of Sag-tension for Overhead Power Lines - Seite 11 →
    # Verifiziert
    durchhang(h=28000, w=15.97, s=300)
    # Beispielrechnung gemäss Cigre 324 WG B2-12 Calculation of Sag-tension for Overhead Power Lines - Seite 12 →
    # Verifiziert
    leiterlaenge(h=28000, w=15.97, s=300)
    # Beispielrechnung gemäss Cigre 324 WG B2-12 Calculation of Sag-tension for Overhead Power Lines - Seite 12 →
    # Verifiziert
    leiterlaenge_spannweite_differenz(h=28000, w=15.97, s=300)
    # Beispielrechnung gemäss Cigre 324 WG B2-12 Calculation of Sag-tension for Overhead Power Lines - Seite 15 →
    # Verifiziert
    seilzugspannung_gesamt_approx(h=28000, w=15.97, d=6.42)
    # Beispielrechnung gemäss Cigre 324 WG B2-12 Calculation of Sag-tension for Overhead Power Lines - Seite 15 →
    # Verifiziert
    seilzugspannung_gesamt_exact(h=28000, w=15.97, s=300)
    # Beispielrechnung gemäss Cigre 324 WG B2-12 Calculation of Sag-tension for Overhead Power Lines - Seite 15 →
    # Verifiziert
    kettenleiter_schraeg_gesamt(h=28000, w=15.97, s=300, h_diff=10, art="links hoch")
    kettenleiter_schraeg_gesamt(h=28000, w=15.97, s=300, h_diff=10, art="rechts hoch")

    # Test für Bonaduz:
    #kettenleiter_gerade_gesamt(h=11500, w=27.743, s=163)
    #durchhang(h=11500, w=27.743, s=163)

    # Test für Eglisau:
    kettenleiter_gerade_gesamt(h=6500, w=11.059, s=22)
    durchhang(h=6500, w=11.059, s=22)
    #seilzugspannung_gesamt_exact(h=6500, w=11.059, s=22)
