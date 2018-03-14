
class Param:
    def __init__(self):
        self.z = None
        self.f = None
        self.i = None
        self.c_bar = None
        self.c = None
        self.o = None
        self.h = None
        self.c_prev = None

        """
        self.v = None  # parameter value
        self.d = None  # derivative
        self.m = None  # momentum for AdaGrad

    def update_value(self, value, is_args_updated=True):
        self.v = value

        if is_args_updated:
            self.d = np.zeros_like(value)  # derivative
            self.m = np.zeros_like(value)  # momentum for AdaGrad

"""