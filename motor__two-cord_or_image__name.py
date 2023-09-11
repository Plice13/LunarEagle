from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u

#POTŘEBOVAL BYCH NĚJAK SPRÁVNĚ UDĚLAT TOHLE VNOŘENÍ DO SUBSLOŽEK, VŮBEC NEVÍM JESTLI TO TAK JE SPRÁVNĚ
import bordel.API.client3 as client


#set motors parameters
number_of_zubu_osa_Azimuth = 8192
number_of_zubu_motor_Azimuth = 16
speed_of_motor_Azimuth = 64 #RPS rotation per second

number_of_zubu_osa_height = 1024
number_of_zubu_motor_height = 16
speed_of_motor_height = 64 #RPS rotation per second

#get Azimuth and height (actually it's right ascension and declination) from astronomical database Simbad
def find_star(name_of_star):
    result_table = Simbad.query_object(name_of_star)
    #print(result_table)
    
    if result_table is not None:
        ra_str = result_table['RA'][0]
        dec_str = result_table['DEC'][0]

        # Convert from sexagesimal format to degrees
        coords = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))

        print(result_table['MAIN_ID'][0], "má souřadnice: ", coords.ra.deg, coords.dec.deg)
        return coords.ra.deg, coords.dec.deg
    else:
        return None, None

def calculate_movement(observed_tuple, target_name):
    Azimuth_target, height_target = find_star(target_name)
    Azimuth_observed, height_observed = observed_tuple
    
    #first calculate Azimuth, then height
    Azimuth_move = Azimuth_target-Azimuth_observed
    print("Stativ se musí otočit v azimutu o", Azimuth_move, "stupňů doprava")

    move_number_of_zubu_osa_Azimuth = Azimuth_move/360*number_of_zubu_osa_Azimuth
    print("Obě ozubená kola se v azimutu musí otočit o", move_number_of_zubu_osa_Azimuth, "zubů")

    move_number_of_rotations_motor_Azimuth = move_number_of_zubu_osa_Azimuth/number_of_zubu_motor_Azimuth 
    print("Motor musí vykonat v azimutu", move_number_of_rotations_motor_Azimuth, "otáček")
    
    time_for_move_Azimuth = move_number_of_rotations_motor_Azimuth/speed_of_motor_Azimuth
    print("Motor se musí v azimutu otáčet", time_for_move_Azimuth, "sekund")

    #now height
    height_move = height_target-height_observed
    print("Stativ se musí otočit ve výšce o", height_move, "stupňů doprava")

    move_number_of_zubu_osa_height = height_move/360*number_of_zubu_osa_height
    print("Obě ozubená kola se ve výšce musí otočit o", move_number_of_zubu_osa_height, "zubů")

    move_number_of_rotations_motor_height = move_number_of_zubu_osa_height/number_of_zubu_motor_height 
    print("Motor musí vykonat ve výšce", move_number_of_rotations_motor_height, "otáček")
    
    time_for_move_height = move_number_of_rotations_motor_height/speed_of_motor_height
    print("Motor se musí ve výšce otáčet", time_for_move_height, "sekund")


#ask if we want manually set Azimuth_observed, height_observed or we have image either fits
mode = input('MANually or AUTOmatically: ')

#get input values
if mode == 'AUTO':
    #implemented uploading picture
    #get Azimuth_observed, height_observed from image

    #CRVAL1, CRVAL2, scale_degrees = client.run('bordel\Pictures\IMG_1124.JPG')

    #get Azimuth_observed, height_observed from fits file, that mean we already run image processing either we have some fits
    Azimuth_observed, height_observed, scale_degrees = client.get_middle('fits\wcs_automatic.fits')
    
else:
    Azimuth_observed, height_observed = map(int, input("Azimuth_observed, height_observed: ").split(', '))

star_name = input("Star to target: ")

calculate_movement((Azimuth_observed, height_observed),star_name)

