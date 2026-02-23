#!/usr/bin/env python3
"""
FAZA 1: Test Kalibracije Kamere

Testira da li kalibracija kamere radi ispravno.
"""

import numpy as np
import cv2
import glob
import os

def test_calibration():
    """Test camera calibration step by step"""
    
    print("=" * 60)
    print("FAZA 1: TEST KALIBRACIJE KAMERE")
    print("=" * 60)
    
    # 1. Učitaj slike šahovske table
    images = glob.glob('camera_cal/calibration*.jpg')
    print(f"\n1. Pronađeno {len(images)} slika za kalibraciju")
    
    if len(images) == 0:
        print("✗ Nema slika u camera_cal/")
        return False
    
    # 2. Pripremi object points (3D tačke u realnom svetu)
    nx, ny = 9, 6  # Broj unutrašnjih uglova
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    
    objpoints = []  # 3D tačke u realnom svetu
    imgpoints = []  # 2D tačke na slici
    
    print(f"\n2. Tražim uglove šahovske table ({nx}x{ny})...")
    
    successful = 0
    failed = 0
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Pronađi uglove
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            successful += 1
            print(f"  ✓ {os.path.basename(fname)}")
        else:
            failed += 1
            print(f"  ✗ {os.path.basename(fname)} - uglovi nisu pronađeni")
    
    print(f"\n  Uspešno: {successful}/{len(images)}")
    print(f"  Neuspešno: {failed}/{len(images)}")
    
    if successful < 10:
        print("\n✗ Premalo uspešnih slika za kalibraciju (potrebno min 10)")
        return False
    
    # 3. Kalibriši kameru
    print("\n3. Računam kalibraciju...")
    img_size = (gray.shape[1], gray.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    
    if not ret:
        print("✗ Kalibracija nije uspela")
        return False
    
    print("  ✓ Kalibracija uspešna")
    print(f"\n  Camera matrix:\n{mtx}")
    print(f"\n  Distortion coefficients:\n{dist}")
    
    # 4. Sačuvaj kalibraciju
    np.savez('calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print("\n4. ✓ Kalibracija sačuvana u calibration.npz")
    
    # 5. Test na jednoj slici
    print("\n5. Testiram korekciju distorzije...")
    test_img = cv2.imread(images[0])
    undist = cv2.undistort(test_img, mtx, dist, None, mtx)
    
    # Sačuvaj uporednu sliku
    comparison = np.hstack((test_img, undist))
    cv2.imwrite('debug/calibration_test.jpg', comparison)
    print("  ✓ Sačuvano: debug/calibration_test.jpg")
    
    # 6. Test na test slici sa linijama
    print("\n6. Testiram na slici sa pravim linijama...")
    if os.path.exists('test_images/straight_lines1.jpg'):
        test_img = cv2.imread('test_images/straight_lines1.jpg')
        undist = cv2.undistort(test_img, mtx, dist, None, mtx)
        
        comparison = np.hstack((test_img, undist))
        cv2.imwrite('debug/undistort_straight_lines.jpg', comparison)
        print("  ✓ Sačuvano: debug/undistort_straight_lines.jpg")
        print("  → Proveri da li su prave linije ostale prave!")
    
    print("\n" + "=" * 60)
    print("✓ FAZA 1 ZAVRŠENA - Kalibracija radi!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    os.makedirs('debug', exist_ok=True)
    success = test_calibration()
    
    if success:
        print("\n→ Možemo nastaviti sa FAZA 2: Binarna Segmentacija")
    else:
        print("\n✗ Popravi kalibraciju pre nego što nastaviš")
