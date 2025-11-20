from linear_no_manoeuvre import PropagateSatelliteLinearNoMan
from second_order_no_manoeuvre import PropagateSatelliteSecondNoMan

# RUN THE ALGORITHMS FROM HERE
#Input format: number of points, time between points, record data?

def short_arc_case_linear():
    propagator = PropagateSatelliteLinearNoMan(300, 2, True)
    propagator.do_calc()

def sparse_case_linear():
    propagator = PropagateSatelliteLinearNoMan(18, 1000, True)
    propagator.do_calc()

def standard_case_linear():
    propagator = PropagateSatelliteLinearNoMan(1080, 10, True)
    propagator.do_calc()

def short_arc_case():
    propagator = PropagateSatelliteSecondNoMan(300, 2, True)
    propagator.do_calc()

def sparse_case():
    propagator = PropagateSatelliteSecondNoMan(18, 1000, True)
    propagator.do_calc()

def standard_case():
    propagator = PropagateSatelliteSecondNoMan(1080, 10, True)
    propagator.do_calc()

# CHANGE THESE TO RUN EACH TEST
if __name__ == "__main__":
    print("LINEAR TESTS:\n")
    print("Running no manoeuvre...")
    standard_case_linear()

    print("Running short arc case...")
    short_arc_case_linear()

    print("Running sparse case...")
    sparse_case_linear()


    print("SECOND ORDER TESTS:\n")
    print("Running no manoeuvre...")
    standard_case()

    print("Running short arc case...")
    short_arc_case()

    print("Running sparse case...")
    sparse_case()