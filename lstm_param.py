
class Param:
    """
    Contains all the parameters used in a LSTM cell
    """
    def __init__(self):
        self.z = None
        self.f = None
        self.i = None
        self.c_bar = None
        self.c = None
        self.o = None
        self.h = None
        self.c_prev = None
