import math
from typing import Tuple, Any, List

import numpy as np
from numpy import ndarray, dtype
import matplotlib.pyplot as plt


def kettenleiter(h: float, w: float, x: float) -> float:
    """
    Funktion zur Berechnung der Postion y(x) eines Kettenleiters
    h: Horizontale Kraft in N/mm2
    w: Leitergewicht pro Längeneinheit in N/m
    x: Postion x für die y bestimmt werden soll in m
    """
    y = h / w * ((math.cosh((w * x) / h)) - 1)
    return y


def kettenleiter_gesamt(h: float, w: float, s: float) -> tuple[ndarray[Any, dtype[Any]], list[float]]:
    """
    Funktion zur Berechnung des gesamten Kettenleiters
    h: Horizontale Kraft in N/mm2
    w: Leitergewicht pro Längeneinheit in N/m
    s: Spannweite in m
    """
    x = np.arange(-s/2, s/2, 0.001)
    y = []
    for count, item in enumerate(x):
        y_yield = kettenleiter(h, w, item)
        y.append(y_yield)
    kettenleiter_plot(x, y)
    return x, y


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
    plt.ylim(-max(y)*0.5/2, max(y)*2.5/2)
    plt.ylabel("Höhe in m")
    plt.legend(h1, l1)
    #plt.gca().set_aspect('equal')
    #plt.subplots_adjust(bottom=.25, left=.25)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show()


def durchhang(h: float, w: float, s: float) -> float:
    """
    Funktion zur Berechnung des maximalen Durchhangs
    h: Horizontale Kraft in N/mm2
    w: Leitergewicht pro Längeneinheit in N/m
    s: Spannweite in m
    """
    d = h / w * ((math.cosh((w * s) / (2 * h))) - 1)
    print("Maximaler Durchhang in m:", d)
    return d


if __name__ == "__main__":
    # Beispielrechnung gemäss Cigre 324 WG B2-12 Calculation of Sag-tension for Overhead Power Lines - Seite 10 →
    # Verifiziert
    kettenleiter_gesamt(h=28000, w=15.97, s=300)
    # Beispielrechnung gemäss Cigre 324 WG B2-12 Calculation of Sag-tension for Overhead Power Lines - Seite 11 →
    # Verifiziert
    durchhang(h=28000, w=15.97, s=300)

    # Test für Bonaduz:
    kettenleiter_gesamt(h=58982, w=27.743, s=10)
    durchhang(h=58982, w=27.743, s=10)