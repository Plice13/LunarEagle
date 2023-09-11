

if __name__ == '__main__':
    Azimut_observed, height_observed, Azimut_target, height_target, zuby_osy, zuby_motory = input("Azimut_observed, height_observed, Azimut_target, height_target, zuby_osy, zuby_motory: ").split(', ')
    
    #převod textu na čísla
    Azimut_observed = int(Azimut_observed)
    height_observed = int(height_observed)
    Azimut_target = int(Azimut_target)
    height_target = int(height_target)
    zuby_osy = int(zuby_osy)
    zuby_motory = int(zuby_motory)

    number_of_rotations = zuby_osy/zuby_motory * (Azimut_target-Azimut_observed)/360
    print(number_of_rotations)