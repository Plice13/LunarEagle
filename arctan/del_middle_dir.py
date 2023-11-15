import os
import shutil

def kopirovat_obrazky_ze_slozek(hlavni_slozka, cilovy_adresar, pismena):
    for letter in pismena:
        if not os.path.isdir(hlavni_slozka):
            print("Zadaná cesta není platná složka.")
            return

        # Vytvořte cílový adresář pro každé písmeno, pokud neexistuje
        cilova_slozka = os.path.join(cilovy_adresar, letter)
        if not os.path.exists(cilova_slozka):
            os.makedirs(cilova_slozka)

        obsah_slozky = os.listdir(hlavni_slozka)
    
        for polozka in obsah_slozky:
            print(polozka)
            cesta_k_polozce = os.path.join(hlavni_slozka, polozka)
            polozka = list(polozka)
            try:
                if os.path.isdir(cesta_k_polozce) and polozka[2] == letter:
                    # Získání seznamu všech souborů ve složce
                    soubory_ve_slozce = [soubor for soubor in os.listdir(cesta_k_polozce) if os.path.isfile(os.path.join(cesta_k_polozce, soubor))]

                    # Kopírování souborů do cílového adresáře pro dané písmeno
                    for soubor in soubory_ve_slozce:
                        cesta_ke_zdrojovemu_souboru = os.path.join(cesta_k_polozce, soubor)
                        cesta_k_cilovemu_souboru = os.path.join(cilova_slozka, soubor)
                        shutil.copy2(cesta_ke_zdrojovemu_souboru, cesta_k_cilovemu_souboru)
            except Exception as e:
                print(e)
# Zadejte cestu k hlavní složce, cílovému adresáři a seznamu písmen
hlavni_slozka = r'C:\Users\PlicEduard\clasifics\classification_every'
cilovy_adresar = r'C:\Users\PlicEduard\clasifics\classification_end_letters'
#pismena = ['x', 'r', 'a', 's', 'h', 'k']
pismena = ['x','i','o','c']

# Zavolejte funkci pro kopírování obrázků ze složek začínajících na daná písmena
kopirovat_obrazky_ze_slozek(hlavni_slozka, cilovy_adresar, pismena)
