"""Convolution.

This tool accepts comma separated value files (.csv).

This script requires that `pandas` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * fluximport - returns the pandas.DataFrame containing the flux data
"""

import numpy as np
import pandas as pd


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
        Index(['minE', 'maxE', 'numu', 'anti-numu', 'nue', 'anti-nue'
        ...   'truenumu', 'trueantinumu', 'truenue', 'trueantinue'],
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

    return df


print_dict = {
    "numu": r"$\nu_\mu$",
    "antinumu": r"$\overline{\nu}_\mu$",
    "nue": r"$\nu_e$",
    "antinue": r"$\overline{\nu}_e$",
}
