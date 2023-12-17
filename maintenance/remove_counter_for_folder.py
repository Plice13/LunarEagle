import os

def obnov_puvodni_nazvy_slozek(hlavni_slozka):
    if not os.path.isdir(hlavni_slozka):
        print("Zadaná cesta není platná složka.")
        return

    obsah_slozky = os.listdir(hlavni_slozka)

    for polozka in obsah_slozky:
        cesta_k_polozce = os.path.join(hlavni_slozka, polozka)
        if os.path.isdir(cesta_k_polozce):
            # Oddělte číslo a původní název složky
            puvodni_nazev = polozka.split('_', 1)[1]
            # Přejmenujte složku na původní název
            os.rename(cesta_k_polozce, os.path.join(hlavni_slozka, puvodni_nazev))

# Zadejte cestu k hlavní složce, ve které chcete obnovit původní názvy složek
hlavni_slozka = r'c:\Users\PlicEduard\AI2\classes\shko'
obnov_puvodni_nazvy_slozek(hlavni_slozka)
