# Grounded parameters for the journal extension

Every value in `configs/grounded.yaml`, and every demand or backhaul value the extension uses, with its source.
Sources were checked on 2026-10-25. **Status:** *primary* = read in the original document; *secondary* = only
through a report that cites it; *derived* = our own calculation from primary inputs (method given);
*open* = not yet verified.

The idealised model (`configs/default.yaml`, no longer in the paper) differs in four places: UL antenna 18 dBi with 60° (physically inconsistent,
row A3), altitude up to 300 m (above both regulatory limits, rows R1–R2), no receiver noise figure (N1), and a
coverage-only objective with no rate model (N3).

## Antenna

| Id | Value | Source | Location | Status |
|---|---|---|---|---|
| A1 | Element: 8 dBi, 65° (V and H), SLA_V = A_max = 30 dB | 3GPP TR 38.901 V17.0.0 | Table 7.3-1 | primary |
| A2 | 2×2 λ/2 panel of A1 elements: 11.0 dBi, HPBW 44.5° (1×1: 8.0/65°, 3×3: 13.9/32°, 4×4: 16.0/24.8°) | `scripts/antenna_panel_fit.py` | numerical integration of A1 × array factor; gain = directivity − 1.8 dB, the offset that maps the element's computed directivity (9.8 dBi) onto its stated 8 dBi | derived |
| A3 | 18 dBi needs a beam of about 25° (4×4 panel, A2); a 60° beam gives about 8–10 dBi | A2; 41,253 deg² = 4π sr | — | derived (textbook rule D ≈ 41,253/(θ₁θ₂) not checked in Balanis/Kraus: open) |
| A4 | Power conservation, ∫G dΩ ≤ 4π: idealised UL pattern (18 dBi, 60°, −2 dBi floor) radiates 6.2× isotropic (+7.9 dB), so it is not physical; grounded UL 0.67× (−1.7 dB, the same as the A1 element); BH 0.43× | `tests/test_antenna.py` | numerical integration of the parametric pattern | derived |

## Channel and radio

| Id | Value | Source | Location | Status |
|---|---|---|---|---|
| C1 | P_LoS S-curve (a, b) and excess loss (η_LoS, η_NLoS): suburban 4.88, 0.43, 0.1, 21; urban 9.61, 0.16, 1.0, 20; dense urban 12.08, 0.11, 1.6, 23; high-rise 27.23, 0.08, 2.3, 34 dB | Al-Hourani, Kandeepan, Lardner, IEEE WCL 3(6):569–572, 2014 | (a, b) evaluated from Eq. (6) with Tables I–II (not tabulated in the letter); η in the text under Fig. 5 | primary |
| C2 | Aerial-UE LoS and path-loss models (UMa-AV, UMi-AV, RMa-AV), 1.5/10/22.5–300 m | 3GPP TR 36.777 V15.0.0 | Annex B, Tables B-1, B-2 | primary; written for ground BS → aerial UE, not used as baseline |
| N1 | Receiver noise figure: UE 9 dB, BS 5 dB | 3GPP TR 36.777 V15.0.0 | Table A.1-1 | primary |
| N2 | Hotzone/RRH: 24 or 30 dBm per 10 MHz carrier at 2 GHz; 5 dBi omni antenna | 3GPP TR 36.814 V9.2.0 | Table A.2.1.1.2-3 | primary (the 5 dBi row still carries an open editor's note) |
| N3 | Truncated Shannon, DL: α = 0.6, SINR_min = −10 dB, 4.4 bit/s/Hz (reached at 22.0 dB) | 3GPP TR 36.942 V17.0.0 | Annex A.1, Table A.1 | primary (22.0 dB derived) |

## Regulation

| Id | Value | Source | Location | Status |
|---|---|---|---|---|
| R1 | ≤ 400 ft (122 m) AGL, except within 400 ft of a structure | 14 CFR §107.51(b) | eCFR, read 2026-10-25 | primary |
| R2 | Open category ≤ 120 m above the surface; otherwise the specific category needs an operational authorisation | Implementing Reg. (EU) 2019/947 | Art. 4(1)(e), Art. 5(1) | primary (original text; amendments not checked) |

## Demand

| Id | Value | Source | Location | Status |
|---|---|---|---|---|
| D1 | Camp settlements: 45 m²/person incl. household plots (30 m² if communal services sit outside) | Sphere Handbook, 4th ed., 2018 | Shelter & Settlement Standard 2, p. 250–251 | primary |
| D2 | Covered living space ≥ 3.5 m²/person | Sphere Handbook 2018 | Standard 3, p. 254 | primary |
| D3 | VoLTE AMR-WB 12.65 kb/s: 30 kb/s (IPv4), 38 kb/s (IPv6) incl. RTP/UDP/IP | 3GPP TS 26.114 V19.3.0 | §6.2.5.2, Table 6.8; Annex K, Table K.5 | primary |
| D4 | Real-time text 3 kb/s; H.264 video call 192 or 384 kb/s | 3GPP TS 26.114 V19.3.0 | Annex E | primary |
| D5 | Activity factor: urban macro 20 % (10,000 users/km²), dense urban 10 % (25,000/km²), rural 20 % (100/km²) | 3GPP TS 22.261 V19.15.0 | Table 7.1-1 (targets, not measurements) | primary |
| D6 | Extreme rural: 30 kb/s busy-hour mean per user; up to 2 Mb/s DL user-experienced | 3GPP TR 38.913 V19.0.0 | Table 6.1.6-1 | primary |
| D7 | Per-cell user cap: 40 UEs per aerial BS (70 in figure examples) | Hydher et al., Sensors 20(21):6140, 2020 | Table 2, Eq. (1) | primary; vendor caps (64–600 active users) secondary |
| D8 | Post-disaster voice: call attempts 50–60× normal, up to 80–90 % restricted; data much less affected | NTT DOCOMO Annual Report FY2010 | p. 26 | primary |
| D9 | Thailand 2025 earthquake: calls +465 % / +545 %, data +9 % / +13 % | True Corp. network-resilience deck, May 2025 | Voice / data slides | primary (operator) |
| D10 | Milan: 1,349,930 residents on 181.84 km² (7,424/km², 2021); census sections CC BY 4.0 | ISTAT Milan report 2024; ISTAT Basi territoriali | p. 2; legal notes | primary |
| D11 | Telecom Italia activity is scaled by an undisclosed constant k; Nov 1, 2013 – Jan 1, 2014 | Barlacchi et al., Sci. Data 2:150055, 2015 | Methods | primary |
| D13 | Census residents of Milan tiles (2.35 km × 2.35 km, 10 × 10 Telecom Italia squares) at mean-activity percentiles 100/90/75/50/25: 71,932 / 58,185 / 26,848 / 12,637 / 7,352 (13,025 to 1,331 per km²) | ISTAT basi territoriali 2021, Lombardy (R03_21.zip), field POP21 per census section, CC BY 4.0 | `scripts/extract_milan_tiles.py`: area-weighted overlap of sections with each tile in UTM 32N | derived from primary |
| D12 | Mobile phone ownership, population aged 10+: 82 % world, 53 % low-income economies (2025) | ITU, Facts and Figures 2025 | "Mobile phone ownership" page | primary (web page; 0.82 applied to the whole population is our approximation) |

Derived from these (our arithmetic): a 2.35 km Milan crop holds about 41,000 residents at the city mean (D10; superseded
by the per-tile census counts of D13), and a
Gaussian cluster of spread σ holds about π(2σ)²/45 m² people at Sphere density (D1): 700 for σ = 50 m, 2,800 for σ = 100 m.

## Backhaul and gateway (WP5)

| Id | Value | Source | Location | Status |
|---|---|---|---|---|
| B1 | LTE small-cell backhaul, DL busy-time mean / peak: 2×2 10 MHz 17.9 / 93.8 Mb/s; aggregate = max(peak, N × busy mean) | NGMN Small Cell Backhaul Requirements v1.0, 2012 | Fig. 13, §3.4.9 | primary |
| B2 | UAV-to-ground LTE backhaul (2 GHz, 20 MHz): about 65–245 Mb/s depending on ground-station density and height | Galkin, Kibiłda, DaSilva, ICC 2018 | Table I, Fig. 4 | primary (rate read off a figure) |
| B3 | 60 GHz 802.11ad drone mesh: PHY up to 2.3 Gb/s; two aligned drones at 60 m, median UDP throughput about 2.3 Gb/s to 200 m, 1.54 at 240 m, 1.15 at 280 m, 0.96 at 320 m, 0.2 at 360 m (read by eye); no results past 360 m; connectivity beyond 240 m only within ±20° relative yaw. Ground-station UAV: three radios, one facing the ground; flown relay: two horizontal; simulations: three per relay. We take 800 Mb/s per link, a 320 m range and three interfaces. The links are half-duplex (802.11ad is TDD): for the two-way tiers a link carries 400 Mb/s in each direction (`docs/half_duplex.md`). The 800 Mb/s of §4.2 is a two-flow aggregate and the 200 Mb/s there the no-relay baseline's low, so neither is a per-link measurement. | Sheshadri et al., SkyHAUL, ACM MobiHoc 2021 (doi:10.1145/3466772.3467039; same plot as Fig. 4 of arXiv 2006.11307v1) | §2.1, Fig. 3b, Fig. 4a; §4.1 (radios); §5 ("3 radios per relay UAV") | primary |
| B4 | Flying COW: fiber in the tether, then satellite transport | Light Reading 2017; NPR 2017 | — | secondary |
| B5 | Wing Loong-2H (Henan 2021): Ku-band satellite backhaul, about 3,600 users; no backhaul rate published | APSTAR news 2021; Critical Comms 2021 | — | primary / secondary |
| B6 | Starlink, US 2H 2025: median DL ≥ 100 Mb/s in every state but Alaska (national median 133.8 Mb/s); median UL ≥ 20 Mb/s in 22 states (Nebraska highest, 24.94 Mb/s); national median UL 21.46 Mb/s and DL 127.39 Mb/s in Q1 2026 (read in the article, 2026-11-04). We use 100 Mb/s down (met in every state but Alaska) and 20 Mb/s up (about the median state); two-way tiers (text, voice, video) are capped by the upload, broadband by the download (`journal_data.satellite_cap_pct`) | S. Marek, "Starlink Hits New Highs in the U.S.," Ookla, 5 May 2026, ookla.com/articles/starlink-hits-new-us-highs | — | primary (measurement, not a specification) |
| B7 | Relay node: 30 dBm, 5 dBi omni, receiver noise figure 5 dB | 3GPP TR 36.814 V9.2.0 | Table A.2.1.1.4-3 | primary (used for the drone backhaul radio; the 4×4 panel of A2 replaces the omni antenna) |

## Open items
- Balanis/Kraus equation for D ≈ 41,253/(θ₁θ₂) (A3), only needed if the paper quotes the rule.
- GEOSTAT 1 km grid licence (ISTAT census sections are CC BY 4.0 and suffice).
- Any measured drone-to-drone Wi-Fi throughput (B-rows currently cover LTE and 60 GHz).
- No standard defines a disaster service tier: the tiers are assembled from D3–D6 and must be presented as such.
