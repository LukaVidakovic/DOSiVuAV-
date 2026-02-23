# Plan: Detekcija Linija Trake - Od Nule do Rešenja

## Cilj
Napraviti robustan sistem za detekciju linija trake sa snimaka kamere iz automobila.

## Faze Razvoja

### FAZA 1: Kalibracija Kamere ✓
**Cilj:** Ukloniti optičku distorziju kamere

**Koraci:**
1. Učitati slike šahovske table iz `camera_cal/`
2. Detektovati uglove šahovske table (9x6)
3. Izračunati matricu kamere i koeficijente distorzije
4. Sačuvati u `calibration.npz`
5. **Test:** Primeniti na test sliku i vizuelno proveriti

**Kriterijum uspeha:** Prave linije na slici ostaju prave nakon korekcije

---

### FAZA 2: Binarna Segmentacija (Thresholding)
**Cilj:** Izdvojiti piksele koji pripadaju linijama

**Koraci:**
1. Testirati Sobel X gradijent na jednoj slici
2. Testirati HLS S-kanal (žute linije)
3. Testirati HLS L-kanal (bele linije)
4. Testirati LAB B-kanal (žute linije)
5. Kombinovati pragove
6. **Test:** Vizuelno proveriti na 3-4 različite slike (bela/žuta, svetlo/senka)

**Kriterijum uspeha:** Jasno vidljive linije na binarnoj slici, minimalan šum

---

### FAZA 3: Perspektivna Transformacija
**Cilj:** Transformisati sliku u "bird's eye view"

**Koraci:**
1. Odabrati source tačke (trapez na slici sa pravim linijama)
2. Definisati destination tačke (pravougaonik)
3. Izračunati transformacionu matricu
4. Primeniti warp
5. **Test:** Prave linije moraju biti paralelne i vertikalne

**Kriterijum uspeha:** Linije su paralelne na warped slici

---

### FAZA 4: Detekcija Piksela Linija
**Cilj:** Pronaći piksele koji pripadaju levoj i desnoj liniji

**Koraci:**
1. Implementirati histogram za pronalaženje početnih pozicija
2. Implementirati sliding window algoritam
3. Sakupiti piksele za levu i desnu liniju
4. **Test:** Vizualizovati prozore i detektovane piksele

**Kriterijum uspeha:** Detektovano >1000 piksela po liniji

---

### FAZA 5: Fitovanje Polinoma
**Cilj:** Fitovati glatke krive kroz detektovane piksele

**Koraci:**
1. Fitovati polinom 2. reda: x = Ay² + By + C
2. Generisati tačke za crtanje
3. **Test:** Vizualizovati fitovane krive preko piksela

**Kriterijum uspeha:** Krive glatko prate linije

---

### FAZA 6: Računanje Zakrivljenosti i Pozicije
**Cilj:** Izračunati metrike puta

**Koraci:**
1. Konvertovati piksele u metre
2. Refitovati polinome u metričkom prostoru
3. Izračunati radijus zakrivljenosti
4. Izračunati poziciju vozila
5. **Test:** Proveriti da li vrednosti imaju smisla (npr. prave linije > 1000m)

**Kriterijum uspeha:** Razumne vrednosti za zakrivljenost i poziciju

---

### FAZA 7: Vizualizacija i Overlay
**Cilj:** Nacrtati detektovanu traku na originalnoj slici

**Koraci:**
1. Nacrtati poligon između linija na warped slici
2. Unwarp nazad na originalnu perspektivu
3. Overlay preko originalne slike
4. Dodati tekst sa metrikama
5. **Test:** Vizuelno proveriti da traka prati linije

**Kriterijum uspeha:** Zelena traka precizno pokriva prostor između linija

---

### FAZA 8: Pipeline za Slike
**Cilj:** Integrisati sve korake u jedan pipeline

**Koraci:**
1. Kreirati klasu `LaneDetectionPipeline`
2. Implementirati `process_image()` metod
3. **Test:** Procesirati sve test slike
4. Analizirati neuspehe

**Kriterijum uspeha:** >90% test slika uspešno procesiranih

---

### FAZA 9: Pipeline za Video
**Cilj:** Optimizovati za video procesiranje

**Koraci:**
1. Implementirati search around poly (brža detekcija)
2. Dodati temporal smoothing (prosek preko frejmova)
3. Dodati sanity checks (validacija detekcije)
4. **Test:** Procesirati kratki video (5-10 sekundi)

**Kriterijum uspeha:** Stabilna detekcija bez trzanja

---

### FAZA 10: Finalizacija
**Cilj:** Dokumentacija i finalni testovi

**Koraci:**
1. Procesirati sve test slike
2. Procesirati sve test videe
3. Napraviti vizualizacije za dokumentaciju
4. Napisati README.md sa objašnjenjima
5. Kreirati primere za docs/

**Kriterijum uspeha:** Kompletan projekat spreman za predaju

---

## Trenutni Status

- [x] FAZA 1: Kalibracija kamere ✓
- [x] FAZA 2: Binarna segmentacija ✓
- [ ] FAZA 3: Perspektivna transformacija
- [ ] FAZA 4: Detekcija piksela
- [ ] FAZA 5: Fitovanje polinoma
- [ ] FAZA 6: Zakrivljenost i pozicija
- [ ] FAZA 7: Vizualizacija
- [ ] FAZA 8: Pipeline za slike
- [ ] FAZA 9: Pipeline za video
- [ ] FAZA 10: Finalizacija

---

## Napomene

- Posle svake faze OBAVEZNO testirati pre nego što nastavimo
- Koristiti debug/ folder za sve testove i vizualizacije
- Ako nešto ne radi, vratiti se korak unazad
- Fokus na JEDNOSTAVNOST i ROBUSNOST, ne na kompleksnost
