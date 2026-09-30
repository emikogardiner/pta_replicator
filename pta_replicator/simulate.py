"""
Code to make simulated PTA datasets with PINT
Created by Bence Becsy, Jeff Hazboun, Aaron Johnson
With code adapted from libstempo (Michele Vallisneri)

"""
import glob
import os
from dataclasses import dataclass
from astropy.time import TimeDelta
from astropy.time.core import Time
import astropy.units as u
import numpy as np

from pint.residuals import Residuals
import pint.toa as toa
from pint import models
from pint.simulation import make_fake_toas_fromMJDs
import pint.fitter
import tqdm

from enterprise.pulsar import Pulsar



@dataclass
class SimulatedPulsar:
    """
    Class to hold properties of a simulated pulsar
    """
    ephem: str = 'DE440'
    model: models.TimingModel = None
    toas: toa.TOAs = None
    residuals: Residuals = None
    name: str = None
    loc: dict = None
    added_signals: dict = None
    added_signals_time: dict = None

    def __repr__(self):
        return f"SimulatedPulsar({self.name})"

    def update_residuals(self):
        """Method to take the current TOAs and model and update the residuals with them"""
        self.residuals = Residuals(self.toas, self.model)

    def fit(self, fitter='auto', **fitter_kwargs):
        """
        Refit the timing model and update everything

        Parameters
        ----------
        fitter : str
            Type of fitter to use [auto]
        fitter_kwargs :
            Kwargs to pass onto fit_toas. Can be useful to set parameters such as max_chi2_increase, min_lambda, etc.
        """
        if fitter == 'wls':
            self.f = pint.fitter.WLSFitter(self.toas, self.model)
        elif fitter == 'gls':
            self.f = pint.fitter.GLSFitter(self.toas, self.model)
        elif fitter == 'downhill':
            self.f = pint.fitter.DownhillGLSFitter(self.toas, self.model)
        elif fitter == 'auto':
            self.f = pint.fitter.Fitter.auto(self.toas, self.model)
        else:
            err = f"{fitter=} must be one of 'wls', 'gls', 'downhill' or 'auto'"
            raise ValueError(err)
        
        self.f.fit_toas(**fitter_kwargs)
        self.model = self.f.model
        self.update_residuals()

    def write_partim(self, outpar: str, outtim: str, tempo2: bool = False):
        """Format for either PINT or Tempo2"""
        self.model.write_parfile(outpar)
        if tempo2:
            self.toas.write_TOA_file(outtim, format='Tempo2')
        else:
            self.toas.write_TOA_file(outtim)

    def update_added_signals(self, signal_name, param_dict, dt=None):
        """
        Update the timing model with a new signal
        """
        if self.added_signals is None: #this is first set to None and then to empty dict if make_ideal() is called - so good way to check if that was done
            raise ValueError("make_ideal() must be called on SimulatedPulsar before adding new signals.")
        if signal_name in self.added_signals:
            raise ValueError(f"{signal_name} already exists in the model.")
        self.added_signals[signal_name] = param_dict
        if dt is not None:
            self.added_signals_time[signal_name] = dt

    def to_enterprise(self, ephem='DE440'):
        """
        Convert to enterprise PintPulsar object
        """
        return Pulsar(self.toas, self.model, ephem=ephem, timing_package='pint')


def simulate_pulsar(parfile: str, obstimes, toaerr, freq=1440.0, observatory="AXIS", flags=None, ephem:str = 'DE440') -> SimulatedPulsar:
    """
    Create a SimulatedPulsar object from a par file and a list of toas

    Parameters
    ----------
    parfile : str
        Path to par file
    obstimes : array
        List of observation times [MJD]
    toaerr : float or array
        Measurement error - either a common error, or a list of errors of the same length as obstimes [us]
    freq : float or array, optional
        Observation frequency - either a common value or a list [MHz]
    observatory : str, optional
        Observatory for fake toas
    """
    if not os.path.isfile(parfile):
        raise FileNotFoundError("par file does not exist.")

    model = models.get_model(parfile)
    toas = make_fake_toas_fromMJDs(obstimes, model,
                                   freq=freq * u.MHz,
                                   obs=observatory,
                                   flags=flags,
                                   error=toaerr * u.us)
    residuals = Residuals(toas, model)
    name = model.PSR.value

    if hasattr(model, 'RAJ') and hasattr(model, 'DECJ'):
        loc = {'RAJ': model.RAJ.value, 'DECJ': model.DECJ.value}
    elif hasattr(model, 'ELONG') and hasattr(model, 'ELAT'):
        loc = {'ELONG': model.ELONG.value, 'ELAT': model.ELAT.value}
    else:
        raise AttributeError("No pulsar location information (RAJ/DECJ or ELONG/ELAT) in parfile.")
    

    return SimulatedPulsar(ephem=ephem, model=model, toas=toas, residuals=residuals, name=name, loc=loc)


def load_pulsar(parfile: str, timfile: str, ephem:str = 'DE440') -> SimulatedPulsar:
    """
    Load a SimulatedPulsar object from a par and tim file

    Parameters
    ----------
    parfile : str
        Path to par file
    timfile : str
        Path to tim file
    """
    if not os.path.isfile(parfile):
        raise FileNotFoundError("par file does not exist.")
    if not os.path.isfile(timfile):
        raise FileNotFoundError("tim file does not exist.")

    model = models.get_model(parfile)
    toas = toa.get_TOAs(timfile, ephem=ephem, planets=True)
    residuals = Residuals(toas, model)
    name = model.PSR.value

    if hasattr(model, 'RAJ') and hasattr(model, 'DECJ'):
        loc = {'RAJ': model.RAJ.value, 'DECJ': model.DECJ.value}
    elif hasattr(model, 'ELONG') and hasattr(model, 'ELAT'):
        loc = {'ELONG': model.ELONG.value, 'ELAT': model.ELAT.value}
    else:
        raise AttributeError("No pulsar location information (RAJ/DECJ or ELONG/ELAT) in parfile.")
    

    return SimulatedPulsar(ephem=ephem, model=model, toas=toas, residuals=residuals, name=name, loc=loc)


def load_from_directories(pardir: str, timdir: str, ephem:str = 'DE440', num_psrs: int = None, debug=False) -> list:
    """
    Takes a directory of par files and a directory of tim files and
    loads them into a list of SimulatedPulsar objects
    """
    if not os.path.isdir(pardir):
        raise FileNotFoundError("par directory does not exist.")
    if not os.path.isdir(timdir):
        raise FileNotFoundError("tim directory does not exist.")
    unfiltered_pars = sorted(glob.glob(pardir + "/*.par"))
    filtered_pars = [p for p in unfiltered_pars if ".t2" not in p]
    unfiltered_tims = sorted(glob.glob(timdir + "/*.tim"))
    combo_list = list(zip(filtered_pars, unfiltered_tims))
    psrs = []
    for par, tim in combo_list:
        if num_psrs:
            if len(psrs) >= num_psrs:
                break
        if debug: print(f"loading {par=}, {tim=}")
        psrs.append(load_pulsar(par, tim, ephem=ephem))
    return psrs


def make_ideal(psr: SimulatedPulsar, iterations: int = 2):
    """
    Takes a pint.TOAs and pint.TimingModel object and effectively zeros out the residuals.
    """
    for ii in range(iterations):
        residuals = Residuals(psr.toas, psr.model)
        psr.toas.adjust_TOAs(TimeDelta(-1.0*residuals.time_resids))
    psr.added_signals = {}
    psr.added_signals_time = {}
    psr.update_residuals()



def generate_new_toas(old_mjds, old_errors, start_mjd, end_mjd, drop_extrema = True):
    """ Generate new observation times (MJDs) and errors, drawn from the previous intervals
    between observation and previous errors. 

    Parameters
    ----------
    old_mjds : astropy.units.quantity.Quantity object
        mjds of all past TOAs, in days 
    old_errors : astropy.units.quantity.Quantity object
        the errors of all past TOAs, usually in us
    start_mjd : float or astropy.time.core.Time object
        starting observation (old toas.last_MJD) float, in days
    end_mjd : float or astropy.time.core.Time
        new final observation date, in days
    drop_extrema : bool
        Whether or not to drop the lowest and highest observing time intervals

    Returns
    -------
    new_mjds : astropy.units.quantity.Quantity
        TOA MJD of each new observation, in days
    new_errors : astropy.units.quantity.Quantity
        TOA error of each new observation, in same units as old_errors, usually us

    """

    # get start and end as floats
    if isinstance(end_mjd, Time):
        end_mjd = end_mjd.value
    if isinstance(start_mjd, Time):
        start_mjd = start_mjd.value

    old_mjd_intervals = np.diff(old_mjds).value
    if drop_extrema:
        old_mjd_intervals = np.sort(old_mjd_intervals)[1:-1] # drop longest and shortest

    new_mjds = []
    new_errors = []
    # generate a new observation, with a time difference from the current one drawn from previous time differences
    cur_mjd = start_mjd + np.random.choice(old_mjd_intervals) 
    # draw the error for the new observation from previous errors
    cur_error = np.random.choice(old_errors)
    
    while (cur_mjd < end_mjd): # if this new observation is before our end date
        # append the current observation
        new_mjds.append(cur_mjd)
        new_errors.append(cur_error)
        # generate next observation
        cur_mjd = cur_mjd + np.random.choice(old_mjd_intervals)
        cur_error = np.random.choice(old_errors)
    new_errors = new_errors * old_errors.unit
    new_mjds = new_mjds * old_mjds.unit
    return new_mjds, new_errors


def extend_pulsar_duration(psr: SimulatedPulsar, end_mjd=None, extend_by_mjd=None):
    """ Generate new toas and add them to the pulsar, until some end date.

    Parameters
    ----------
    psr : pta_replicator.simulate.SimulatedPulsar object
        simulated pulsar, to have toas added to it
    end_mjd : float or Time object
        new date to end observations, in MJDs
    extend_by_mjd : float or Time object
        number of MJDs to extend beyond latest
    
    Only provide one of end_mjd or extend_by_mjd
    The new residuals will be wacky until you make ideal and re-inject noise.

    
    """
    if end_mjd is not None and extend_by_mjd is not None:
        err = f"Only provide one of {end_mjd=} and {extend_by_mjd=}"
        raise ValueError(err)
    elif end_mjd is None:
        if extend_by_mjd is None:
            err = f"Must provide one of {end_mjd=} or {extend_by_mjd=}"
            raise ValueError(err)
        else:
            end_mjd = psr.toas.last_MJD.value + extend_by_mjd

    # get new mjds and errors
    new_mjds, new_errors = generate_new_toas(
        old_mjds=psr.toas.get_mjds(), old_errors=psr.toas.get_errors(),
        start_mjd=psr.toas.last_MJD, end_mjd=end_mjd)
    
    # make the TOA objects
    new_toas = toa.get_TOAs_array(
        new_mjds, obs=list(psr.toas.observatories)[0], errors=new_errors,
        planets=True, ephem=psr.ephem)
    
    # combine the new custom toas with the old ones 
    new_toas.obliquity = psr.toas.obliquity # use same obliquity ig
    psr.toas.merge(new_toas)

    # update the residuals to include all the new ones
    psr.update_residuals()

def shorten_pulsar_duration(psr: SimulatedPulsar, end_mjd, make_new = False, debug=False):
    """ Remove toas that are after the end date.

    Parameters
    ----------
    psr : pta_replicator.simulate.SimulatedPulsar object
        simulated pulsar, to have toas added to it
    end_mjd : float or Time object
        MJD for new end of observations, in days
    make_new : boolean
        Whether to return a new pulsar, otherwise modify current
    debug : string
        Whether to print debugging info
    
    Only provide one of end_mjd or extend_by_mjd
    The new residuals will be wacky until you make ideal and re-inject noise.

    
    """

    if isinstance(end_mjd, Time):
        end_mjd = end_mjd.value

    cur_first_mjd = psr.toas.first_MJD.value
    cur_last_mjd = psr.toas.last_MJD.value

    if cur_first_mjd > end_mjd:
        err = f"Pulsar data starts at {cur_first_mjd}, no data before {end_mjd=}."
        raise ValueError(err)
    elif cur_last_mjd <= end_mjd:
        if debug: print(f"{cur_last_mjd=} already before {end_mjd=}")
        return
    
    if make_new:
        toas = psr.toas[psr.toas.get_mjds() <= (end_mjd*u.d)]
        model = psr.model
        residuals = Residuals(toas, model)
        new_psr = SimulatedPulsar(
            ephem=psr.ephem, model=model, 
            toas=toas, residuals=residuals, name=psr.name, loc=psr.loc)
        return new_psr

    else:
        # limit the toas 
        psr.toas = psr.toas[psr.toas.get_mjds() <= (end_mjd*u.d)]
        # update the residuals to include all the new ones
        psr.update_residuals()
        return 