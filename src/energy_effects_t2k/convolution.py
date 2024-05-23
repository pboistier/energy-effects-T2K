"""Convolution

This tool accepts comma separated value files (.csv).

This script requires that `pandas` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * fluximport - returns the pandas.DataFrame containing the flux data
"""

import numpy as np
import pandas as pd
import scipy.signal as sig


def fluximport(name: str) -> pd.DataFrame:
    """
    Creates a ``DataFrame`` with the flux data.

    Transforms a (.csv) file from the T2K flux public release into a `pandas`
    ``DataFrame``, by adding some new columns, like the integrated flux per
    bin.
    The units of energies should be in GeV.
    The units of fluxes should be in (cm^2 (50 MeV) (1e21 PoT))^{-1}.

    Parameters
    ----------
    name : str
        The name of the .csv file. Its columns should at least contain
        {`minE`, `maxE`, `numu`, `anti-numu`, `nue`, `anti-nue`} or
        {`minE`, `maxE`, `numu`, `antinumu`, `nue`, `antinue`}.


    Returns
    -------
    pd.DataFrame
        Dataframe which columns are

        >>> df = fluximport(name='some_csv_file.csv')
        Index(['minE', 'maxE', 'numu', 'antinumu', 'nue', 'antinue',
        ...   'truenumu', 'trueantinumu', 'truenue', 'trueantinue',
        ...   'total', 'truetotal'],
        ...   dtype='float')
    """

    df = pd.read_csv(
        name,
        delimiter=",",
        usecols=(lambda x: x != "Bin#"),
    )

    df = df.rename(columns={"anti-numu": "antinumu", "anti-nue": "antinue"})

    for key in ["numu", "nue", "antinumu", "antinue"]:
        df["true" + key] = df[key] * (df.maxE - df.minE) * 1e3 / (50 * 1e21)

    df["total"] = df.numu + df.nue + df.antinumu + df.antinue
    df["truetotal"] = df.truenumu + df.truenue + df.trueantinumu + df.trueantinue

    return df


print_dict = {
    "numu": r"$\nu_\mu$",
    "antinumu": r"$\overline{\nu}_\mu$",
    "nue": r"$\nu_e$",
    "antinue": r"$\overline{\nu}_e$",
    "nutau": r"$\nu_\tau$",
    "antinutau": r"$\overline{\nu}_\tau$",
}


def xsecimport(
    name: str,
    mode: str | list[str] = "raw",
    on=None,
    keep_left: int = -1,
    keep_right: int = 0,
) -> pd.DataFrame:
    df = pd.read_csv(name, delimiter=";", names=["energy", "xsec"])

    if "true" in mode:
        df.xsec = df.xsec * df.energy * 1e-38

    if "filter" in mode:
        cachedf = pd.DataFrame()
        cachedf["energy"] = sig.savgol_filter(
            x=df.energy, window_length=10, polyorder=2, mode="interp"
        )
        cachedf["xsec"] = sig.savgol_filter(
            x=df.xsec, window_length=12, polyorder=1, mode="interp"
        )
        for key in ["energy", "xsec"]:
            cachedf.loc[0:keep_left, key] = df.loc[0:keep_left, key]
            cachedf.loc[keep_right:-1, key] = df.loc[keep_right:-1, key]
        df = cachedf

    if "interp" in mode:
        if on is None:
            raise AttributeError("Missing xdata to interpolate onto.")
        else:
            cachedf = pd.DataFrame()
            cachedf["energy"] = on
            cachedf["xsec"] = np.interp(x=on, xp=df.energy, fp=df.xsec, left=0, right=0)
            df = cachedf

    return df


def energyimport(
    name: str,
    mode: str | list[str] = "raw",
    on=None,
    keep_left: int = -1,
    keep_right: int = 0,
) -> pd.DataFrame:
    df = pd.read_csv(name, delimiter=";", names=["relatenergy", "events"])

    if "true" in mode:
        df.xsec = df.xsec * df.energy * 1e-38

    if "filter" in mode:
        cachedf = pd.DataFrame()
        cachedf["energy"] = sig.savgol_filter(
            x=df.energy, window_length=10, polyorder=2, mode="interp"
        )
        cachedf["xsec"] = sig.savgol_filter(
            x=df.xsec, window_length=12, polyorder=1, mode="interp"
        )
        for key in ["energy", "xsec"]:
            cachedf.loc[0:keep_left, key] = df.loc[0:keep_left, key]
            cachedf.loc[keep_right:-1, key] = df.loc[keep_right:-1, key]
        df = cachedf

    if "interp" in mode:
        if on is None:
            raise AttributeError("Missing xdata to interpolate onto.")
        else:
            df.loc[0] = [-0.5, 0]
            df.loc[-1] = [1.0, 0]
            cachedf = pd.DataFrame()
            cachedf["relatenergy"] = on
            cachedf["events"] = np.interp(x=on, xp=df.relatenergy, fp=df.events)
            df = cachedf

    return df


"""
def convolve(value: tuple, gate, bin) -> list:


    return
"""
