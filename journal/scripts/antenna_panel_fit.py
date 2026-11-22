"""Gain and -3 dB beamwidth of square panels of TR 38.901 antenna elements.

Each element follows TR 38.901 Table 7.3-1 (8 dBi, 65 deg, SLA_V = A_max = 30 dB);
the elements sit lambda/2 apart and the panel steers broadside. Prints the
numerically integrated directivity and HPBW, and the gain after subtracting
the offset that maps the single element's directivity onto its stated 8 dBi.
configs/grounded.yaml uses the 2x2 row (docs/parameters.md, rows A1-A2).

python scripts/antenna_panel_fit.py
"""
import numpy as np

th = np.radians(np.linspace(0, 180, 1441))   # angle from boresight
ph = np.radians(np.linspace(0, 360, 1441, endpoint=False))
T, P = np.meshgrid(th, ph, indexing='ij')
# element: boresight along +x of a local frame; vertical cut angle from boresight measured in elevation, horizontal in azimuth
# use direction cosines: boresight u = cos T; elevation-like angle and azimuth-like angle
x = np.cos(T); y = np.sin(T)*np.cos(P); z = np.sin(T)*np.sin(P)
el = np.degrees(np.arcsin(np.clip(z, -1, 1)))          # vertical offset from boresight
az = np.degrees(np.arctan2(y, x))                        # horizontal offset
AV = -np.minimum(12*(el/65)**2, 30); AH = -np.minimum(12*(az/65)**2, 30)
Adb = -np.minimum(-(AV+AH), 30)
E = 10**(Adb/10)                                        # relative element power pattern
dOm = np.sin(T)*(th[1]-th[0])*(ph[1]-ph[0])
def stats(n):
    k = np.arange(n) - (n-1)/2
    # array in y-z plane, phase = pi*(m*y + l*z) for lambda/2
    AFy = np.abs(np.exp(1j*np.pi*np.multiply.outer(y, k)).sum(-1))**2
    AFz = np.abs(np.exp(1j*np.pi*np.multiply.outer(z, k)).sum(-1))**2
    Pw = E*AFy*AFz
    D = 4*np.pi*Pw.max()/np.sum(Pw*dOm)
    # HPBW in the phi=0 cut
    cut = Pw[:, 0]/Pw[:, 0].max()
    hp = np.degrees(th[np.argmax(cut < 0.5)])*2
    return 10*np.log10(D), hp
def panel_rows(sizes=(1, 2, 3, 4)):
    """[(n, directivity dBi, gain dBi, HPBW deg)]; gain = directivity - (1x1 directivity - 8 dBi)."""
    rows = [(n, *stats(n)) for n in sizes]
    offset = stats(1)[0] - 8.0  # TR 38.901 states 8 dBi for one element
    return [(n, d, d - offset, hp) for n, d, hp in rows]


if __name__ == "__main__":
    for n, d, g, hp in panel_rows():
        print(f"{n}x{n}: directivity {d:5.1f} dBi, gain {g:5.1f} dBi, HPBW {hp:5.1f} deg")
