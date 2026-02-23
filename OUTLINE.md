# Detekcija Linija Trake - Lane Detection

Projekat za detekciju linija na putu korišćenjem OpenCV biblioteke.

## Rezultati

### Slike
- **Uspešnost: 17/17 (100%)**
- Sve test slike uspešno procesovane
- Rezultati u `output_images/`

### Video
- **project_video01-03: 100% uspešnost**
- **challenge01-03: 98-100% uspešnost**
- Rezultati u `output_videos/`

## Korišćenje

### Instalacija
```bash
pip install -r requirements.txt
```

### Kalibracija kamere
```bash
python debug/test_faza1_calibration.py
```

### Procesiranje slika
```bash
python lane_detection_final.py --images
```

### Procesiranje videa
```bash
python lane_detection_final.py --videos
```

### Procesiranje svega
```bash
python lane_detection_final.py
```

## Pipeline Koraci

1. **Kalibracija kamere** - Uklanjanje distorzije
2. **Binarna segmentacija** - Detekcija žutih i belih linija
3. **Perspektivna transformacija** - Bird's eye view
4. **Sliding window** - Pronalaženje piksela linija
5. **Polynomial fitting** - Fitovanje krive
6. **Metrike** - Zakrivljenost i pozicija vozila
7. **Vizualizacija** - Overlay na originalnoj slici

## Struktura

```
DOSiVuAV-/
├── camera_cal/              # Slike za kalibraciju
├── test_images/             # Test slike
├── test_videos/             # Test videi
├── output_images/           # Rezultati - slike
├── output_videos/           # Rezultati - videi
├── debug/                   # Debug skripte
├── calibration.npz          # Kalibracioni parametri
├── lane_detection_final.py  # Glavni pipeline
└── PLAN.md                  # Plan razvoja
```

## Tehnologije

- Python 3.13
- OpenCV 4.x
- NumPy

## Status

✓ Kalibracija kamere  
✓ Binarna segmentacija  
✓ Perspektivna transformacija  
✓ Detekcija piksela  
✓ Fitovanje polinoma  
✓ Računanje metrika  
✓ Vizualizacija  
✓ Pipeline za slike (100%)  
✓ Pipeline za video (98-100%)  

## TODO

- [ ] Poboljšati preklapanje crvene/plave linije sa stvarnim linijama
- [ ] Optimizovati za challenge03 video
- [ ] Dodati temporal smoothing za stabilniju detekciju
