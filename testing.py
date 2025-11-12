#from STTMethod import PropagateSatellite
from linear_no_manoeuvre import PropagateSatelliteLinearNoMan
from second_order_no_manoeuvre import PropagateSatelliteSecondNoMan
from linear_man_BROKEN import PropagateSatelliteLinearMan


# RUN THE ALGORITHMS FROM HERE
#Input format: number of points, time between points, actual dv, t_man actual, initial estimate dv, t_man estimate, record data?

def standard_case():
    propagator = PropagateSatelliteLinearMan(180, 10, 3, 3, 3, 905, 1, 2, 3, 955, True)
    propagator.do_calc()

def short_arc_case():
    propagator = PropagateSatellite(300, 2, 10, 10, 10, 150, 6, 7, 8, 900, True)
    propagator.do_calc()

def sparse_case():
    propagator = PropagateSatellite(18, 100, 3, 3, 3, 900, 1, 2, 3, 150, True)
    propagator.do_calc()

def late_impulse():
    propagator = PropagateSatellite(180, 10, 1, 2, 4, 1600, 3, 2, 1, 1600, True)
    propagator.do_calc() 

def early_impulse():
    propagator = PropagateSatellite(180, 10, 1, 2, 4, 30, 1, 2, 3, 30, True)
    propagator.do_calc()

def no_manoeuvre():
    propagator = PropagateSatelliteLinearNoMan(180, 10, True) # Change this to 3 hours each
    propagator.do_calc()

# CHANGE THESE TO RUN EACH TEST
if __name__ == "__main__":
    # print("Running no manoeuvre...")
    # no_manoeuvre()
    
    print("Running standard case....")
    standard_case()

    # print("Running short arc case...")
    # short_arc_case()

    # print("Running sparse case...")
    # sparse_case()

    # print("Running early impulse case...")
    # early_impulse()

    # print("Running late impulse case...")
    # late_impulse()