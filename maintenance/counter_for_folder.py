import os

def zobraz_pocet_souboru_v_slozkach(hlavni_slozka):
    if not os.path.isdir(hlavni_slozka):
        print("Zadaná cesta není platná složka.")
        return

    obsah_slozky = os.listdir(hlavni_slozka)

    for polozka in obsah_slozky:
        cesta_k_polozce = os.path.join(hlavni_slozka, polozka)
        if os.path.isdir(cesta_k_polozce):
            pocet_souboru = len([name for name in os.listdir(cesta_k_polozce) if os.path.isfile(os.path.join(cesta_k_polozce, name))])
            print(f"Složka {polozka} obsahuje {pocet_souboru} souborů.")
            #os.rename(cesta_k_polozce, os.path.join(hlavni_slozka, str(pocet_souboru)+'_'+polozka))
# Zadejte cestu k hlavní složce, ve které chcete zjistit počet souborů v podadresářích
hlavni_slozka = r'C:\Users\PlicEduard\AI3_full_circle\classes\everything'
zobraz_pocet_souboru_v_slozkach(hlavni_slozka)
