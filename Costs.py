import numpy as np

class Costs(object):
    """
    This class provides cost information for wells depending on target formation:
    Marcellus
    Utica
    ---------------------------------------------------------------------------------
    Provides cost information for each stage of a well construction:
    Pad Construction
    Drilling
    Completion
    Equipment
    Abandonment
    """

    def __init__(self):

        # Marcellus Drilling Equation Parameters
        self.marcellus_fixed_drill = 500000
        self.marcellus_drill_a = 300000
        self.marcellus_drill_b = 0.0001
        self.marcellus_drill_c = 1.02

        # Marcellus Toe Prep Equation Parameters
        self.marcellus_toe_prep_fixed = 50000

        # Marcellus Frac Equation Parameters
        self.marcellus_completion_a = 300.00
        self.marcellus_fixed_completion = 50000

        # Marcellus Drillout Equation Parameters
        self.marcellus_drillout_fixed = 150000
        self.marcellus_drillout_a = 15.00

        # Marcellus Frac Water Equation Parameters
        self.water_per_barrel = 2.50
        self.marcellus_water_per_stage = 5000
        self.marcellus_stage_length = 200

        # Marcellus Flowback Equation Parameters
        self.marcellus_flowback_fixed = 50000

        # Utica Drilling Equation Parameters
        self.utica_fixed_drill = 1500000
        self.utica_drill_a = 500000
        self.utica_drill_b = 0.0001
        self.utica_drill_c = 0.94

        # Utica Toe Prep Equation Parameters
        self.utica_toe_prep_fixed = 100000

        # Utica Frac Equation Parameters
        self.utica_completion_a = 400.00
        self.utica_fixed_completion = 100000

        # Utica Drillout Equation Parameters
        self.utica_drillout_fixed = 200000
        self.utica_drillout_a = 15.0

        # Utica Frac Water Equation Parameters
        self.utica_water_per_stage = 10000
        self.utica_stage_length = 200

        # Utica Flowback Equation Parameters
        self.utica_flowback_fixed = 100000

        # Rig Move Equation Parameters
        self.total_rig_move = 1000000
        self.return_trip_construction = 1000000
        self.new_construction = 3000000

        # Production Equipment Parameters
        self.production_equipment = 300000
        self.fresh_water_injection = 100000

        # Abandonment Cost Parameters
        self.aban = 150000

    def abandonment(self):
        return self.aban

    def construction(self, wells_per_pad=7, return_trip=1):
        if return_trip == 1:
            return self.return_trip_construction / wells_per_pad
        else:
            return self.new_construction / wells_per_pad

    def rig_move(self, wells_per_pad=7):
        return self.total_rig_move / wells_per_pad

    def prod_equipment(self, water_injection=0):
        if water_injection == 1:
            return self.production_equipment + self.fresh_water_injection
        else:
            return self.production_equipment

    # Utica Drilling Equations
    def utica_drill(self, lateral_length=7500):
        return self.utica_fixed_drill + self.utica_drill_a * np.exp(
            lateral_length * self.utica_drill_b) * self.utica_drill_c

    def utica_total_drill(self, lateral_length=7500, wells_per_pad=7, return_trip=1):
        return self.construction(wells_per_pad, return_trip) + self.rig_move(wells_per_pad) + self.utica_drill(
            lateral_length)

    # Utica Completion Equations
    def utica_drillout(self, lateral_length=7500):
        return self.utica_drillout_fixed + self.utica_drillout_a * lateral_length

    def utica_toe_prep(self, lateral_length=7500):
        return self.utica_toe_prep_fixed

    def utica_frac(self, lateral_length=7500):
        return self.utica_fixed_completion + self.utica_completion_a * lateral_length

    def utica_water(self, lateral_length=7500):
        return self.water_per_barrel * self.utica_water_per_stage * (lateral_length // self.utica_stage_length)

    def utica_flowback(self, lateral_length=7500):
        return self.utica_flowback_fixed

    def utica_total_completion(self, lateral_length=7500):
        return self.utica_drillout(lateral_length) + self.utica_toe_prep(lateral_length) + self.utica_frac(
            lateral_length) + \
               self.utica_water(lateral_length) + self.utica_flowback(lateral_length)

    # Marcellus Drilling Equations
    def marcellus_drill(self, lateral_length=7500):
        return self.marcellus_fixed_drill + self.marcellus_drill_a * np.exp(lateral_length * self.marcellus_drill_b) * \
               self.marcellus_drill_c

    def marcellus_total_drill(self, lateral_length=7500, wells_per_pad=7, return_trip=1):
        return self.construction(wells_per_pad, return_trip) + self.rig_move(wells_per_pad) + \
               self.marcellus_drill(lateral_length)

    # Marcellus Completion Equations
    def marcellus_drillout(self, lateral_length=7500):
        return self.marcellus_drillout_fixed + self.marcellus_drillout_a * lateral_length

    def marcellus_toe_prep(self, lateral_length=7500):
        return self.marcellus_toe_prep_fixed

    def marcellus_frac(self, lateral_length=7500):
        return self.marcellus_fixed_completion + self.marcellus_completion_a * lateral_length

    def marcellus_water(self, lateral_length=7500):
        return self.water_per_barrel * self.marcellus_water_per_stage * (lateral_length // self.marcellus_stage_length)

    def marcellus_flowback(self, lateral_length=7500):
        return self.marcellus_flowback_fixed

    def marcellus_total_completion(self, lateral_length=7500):
        return self.marcellus_drillout(lateral_length) + self.marcellus_toe_prep(lateral_length) + \
               self.marcellus_frac(lateral_length) + self.marcellus_water(lateral_length) + self.marcellus_flowback(
            lateral_length)