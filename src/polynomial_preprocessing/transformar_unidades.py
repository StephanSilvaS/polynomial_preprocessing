from astropy.coordinates import Angle
import astropy.units as u
import numpy as np

class TransformarUnidades:
    def __init__(self, ra_value, dec_value):
        self.ra_value = ra_value
        self.dec_value = dec_value
    

    def transform_ra_to_hms(self):
        RA_deg = self.ra_value * (180 / np.pi)

        # Convertir RA a HMS
        ra_hms = Angle(RA_deg * u.deg).to_string(unit=u.hour, sep=':')
        print(f"RA (HMS): {ra_hms}")


    def transform_dec_to_dms(self):
        Dec_deg = self.dec_value * (180 / np.pi)  

        # Convertir Dec a DMS
        dec_dms = Angle(Dec_deg * u.deg).to_string(unit=u.deg, sep=':')
        print(f"Dec (DMS): {dec_dms}")

    def transform_both_units(self):
        
        self.transform_ra_to_hms()
        self.transform_dec_to_dms()
        