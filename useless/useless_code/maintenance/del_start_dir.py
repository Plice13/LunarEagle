import os
import shutil

def kopirovat_obrazky_ze_slozek(hlavni_slozka, cilovy_adresar):
    for letter in ['A','B','C','D','E','F']:
        if not os.path.isdir(hlavni_slozka):
            print("Zadaná cesta není platná složka.")
            return

        # Vytvořte cílový adresář, pokud neexistuje
        cilova_slozka = os.path.join(cilovy_adresar, letter)
        if not os.path.exists(cilova_slozka):
            os.makedirs(cilova_slozka)

        obsah_slozky = os.listdir(hlavni_slozka)
    
        for polozka in obsah_slozky:
            cesta_k_polozce = os.path.join(hlavni_slozka, polozka)
            if os.path.isdir(cesta_k_polozce) and polozka.startswith(letter):
                # Získání seznamu všech souborů ve složce
                soubory_ve_slozce = [soubor for soubor in os.listdir(cesta_k_polozce) if os.path.isfile(os.path.join(cesta_k_polozce, soubor))]

                # Kopírování souborů do cílového adresáře
                for soubor in soubory_ve_slozce:
                    cesta_ke_zdrojovemu_souboru = os.path.join(cesta_k_polozce, soubor)
                    cesta_k_cilovemu_souboru = os.path.join(cilovy_adresar, letter, soubor)
                    shutil.copy2(cesta_ke_zdrojovemu_souboru, cesta_k_cilovemu_souboru)

# Zadejte cestu k hlavní složce a cílovému adresáři
hlavni_slozka = r'C:\Users\PlicEduard\clasifics\classification_every'
cilovy_adresar = r'C:\Users\PlicEduard\clasifics\classification_first_letter'

# Zavolejte funkci pro kopírování obrázků ze složek začínajících na 'A'
kopirovat_obrazky_ze_slozek(hlavni_slozka, cilovy_adresar)
