from astropy.coordinates import HeliographicStonyhurst, solar_system_ephemeris, get_sun
from astropy.time import Time

# Specifikujte datum a čas, pro který chcete získat heliografické souřadnice
observing_time = Time('2023-10-23T12:00:00')  # Změňte datum a čas podle potřeby

# Nastavte sluneční soustavu jako zdroj ephemeridy
with solar_system_ephemeris.set('builtin'):
    sun = get_sun(observing_time)

# Získejte heliografické souřadnice středu Slunce v systému Stonyhurst
heliographic_coords = sun.transform_to(HeliographicStonyhurst)

# Vytiskněte výsledné souřadnice
print(f"Heliografické souřadnice středu Slunce pro {observing_time}:")
print(f"Latitude: {heliographic_coords.lat}")
print(f"Longitude: {heliographic_coords.lon}")
