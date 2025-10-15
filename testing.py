from STTMethod import PropagateSatellite

def standard_case():
    propagator = PropagateSatellite(180, 10, 3, 3, 3, 910, 1, 2, 3, 960, True)
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
    propagator = PropagateSatellite(180, 10, 1, 2, 4, 30, 0, 0, 0, 0, True)
    propagator.do_calc()

if __name__ == "__main__":
    print("Running no manoeuvre...")
    no_manoeuvre()
    
    #print("Running standard case....")
    #standard_case()

    # print("Running short arc case...")
    # short_arc_case()

    # print("Running sparse case...")
    # sparse_case()

    # print("Running early impulse case...")
    # early_impulse()

    # print("Running late impulse case...")
    # late_impulse()