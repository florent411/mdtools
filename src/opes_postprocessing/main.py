

# Contstants
kb = 1.38064852e-23 # Boltzman's constant in m^2 kg s^-2 K^-1
NA = 6.02214086e23 # Avogadro's constant in mol^-1

class OPP():
    
    def __init__(self,
                 input, # Input dataframe from which to calculate the FES
                 type='colvar', # Type of dataframe. Should be 'colvar, state or kernels'.
                 temp=310, # in Kelvin
                 units='kT', # Output units Should be 'kT', 'kJ/mol' or 'kcal/mol'.
                 mintozero=True, # Shift the minimum to zero
                 process_max='last', # Process all frames (all), only last frame (last) or n frames (int).')
                 bins=100, # Number of bins for the grid for each dimension (comma separated). If 1 value is given, it is used for all dimensions.')
                 device='mt', # How to run calculations (mt, np or torch). Mt is a looped code structure that can be multythreaded (fastest for smaller sets). Np performs all calculations using numpy arrays. Torch offloads to GPU (CUDA or MLS, the MacOS Intel GPU). If no GPU is available it runs on the CPUs.
                 save=False, # save FES in backpack
                 **kwargs, # Other options: grid_in, grid_max
                 ):
        

        # Setup all constants
        self.input = input
        self.type = type
        self.temp = temp
        self.units = units
        self.mintozero = mintozero
        self.process_max = process_max
        self.bins = bins
        self.device = device
        self.save = save

        # Energy variables
        if units == 'kJ/mol':
            # 1 kT = 0.008314459 kJ/mol
            unitfactor = kb * NA * self.temp / 1000
        elif units == 'kcal/mol':
            # 1 kJ/mol = 0.238846 kcal/mol
            unitfactor = 0.238846 * kb * NA * self.temp / 1000
        elif units == 'kT':
            # 1 kT = 1 kT
            unitfactor = 1
