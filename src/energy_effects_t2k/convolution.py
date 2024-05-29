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
import src.nu_osc.osc_prob as osc
import src.nu_osc.default_parameters as param


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

    df["nutau"] = 0
    df["antinutau"] = 0

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
) -> pd.DataFrame:

    df = pd.read_csv(name, delimiter=";", names=["relatenergy", "events"])

    if "interp" in mode:
        if on is None:
            raise AttributeError("Missing xdata to interpolate onto.")
        else:
            df = pd.concat(
                [pd.DataFrame({"relatenergy": [-0.5], "events": [0.0]}), df],
                ignore_index=True,
            )
            df = pd.concat(
                [df, pd.DataFrame({"relatenergy": [1.0], "events": [0.0]})],
                ignore_index=True,
            )
            cachedf = pd.DataFrame()
            cachedf["relatenergy"] = on
            cachedf["events"] = np.interp(x=on, xp=df.relatenergy, fp=df.events)
            df = cachedf

    return df


def convolve(
    input: pd.DataFrame,
    gate: pd.DataFrame,
    x_keys: str | list[str] = ["minE", "maxE"],
    y_keys: str | list[str] = [
        "numu",
        "nue",
        "nutau",
        "antinumu",
        "antinue",
        "antinutau",
    ],
) -> pd.DataFrame:

    output = pd.DataFrame()
    for key in x_keys:
        output[key] = input[key]

    Emoy = (input.minE.iloc[0] + input.maxE.iloc[0]) / 2
    true_energy = gate.relatenergy * Emoy + Emoy
    step = input.maxE.iloc[0] - input.minE.iloc[0]
    k_left = int(np.abs(np.trunc((input.minE.iloc[0] - true_energy.iloc[0]) / step)))
    output = pd.concat(
        [
            pd.DataFrame(
                {
                    "minE": [
                        input.minE.iloc[0] + step * n for n in range(-1, -k_left, -1)
                    ],
                    "maxE": [
                        input.maxE.iloc[0] + step * n for n in range(-1, -k_left, -1)
                    ],
                }
            ),
            output,
        ],
        ignore_index=True,
    )

    Emoy = (input.minE.iloc[-1] + input.maxE.iloc[-1]) / 2
    true_energy = gate.relatenergy * Emoy + Emoy
    step = input.maxE.iloc[-1] - input.minE.iloc[-1]
    k_right = (
        int(np.abs(np.trunc((true_energy.iloc[-1] - input.maxE.iloc[-1]) / step))) + 1
    )
    output = pd.concat(
        [
            output,
            pd.DataFrame(
                {
                    "minE": [input.minE.iloc[-1] + step * n for n in range(1, k_right)],
                    "maxE": [input.maxE.iloc[-1] + step * n for n in range(1, k_right)],
                }
            ),
        ],
        ignore_index=True,
    )

    for key in y_keys:
        output[key] = 0.0

    for i in range(len(input)):
        Emoy = (input.minE.iloc[i] + input.maxE.iloc[i]) / 2
        true_energy = gate.relatenergy * Emoy + Emoy

        raw_integral = 0
        raw_integral += gate.events.iloc[0] * (
            true_energy.iloc[1] - true_energy.iloc[0]
        )
        raw_integral += gate.events.iloc[-1] * (
            true_energy.iloc[-1] - true_energy.iloc[-2]
        )
        for k in range(1, len(true_energy) - 1):
            raw_integral += (
                gate.events.iloc[k]
                * (true_energy.iloc[k + 1] - true_energy.iloc[k - 1])
                / 2
            )

        for key in y_keys:
            true_integral = input[key].iloc[i] * (
                input.maxE.iloc[i] - input.minE.iloc[i]
            )
            true_events = gate.events * true_integral / raw_integral

            interpolated = np.interp(
                x=(output.maxE + output.minE) / 2, xp=true_energy, fp=true_events
            )
            interpolated_integral = (interpolated * (output.maxE - output.minE)).sum()
            if interpolated_integral != 0:
                interpolated = interpolated * true_integral / interpolated_integral
            else:
                interpolated = 0

            output[key] += interpolated

    return output


def oscillate(
    to_oscillate: pd.DataFrame,
    **osc_param,
):
    osc_param = {
        "theta_12": param.NUFIT["theta_12"]["bf"],
        "theta_13": param.NUFIT["theta_13"]["bf"],
        "theta_23": param.NUFIT["theta_23"]["bf"],
        "delta_cp": param.NUFIT["delta_CP"]["bf"],
        "dm2_21": param.NUFIT["Dm_square_21"]["bf"],
        "dm2_atm": param.NUFIT["Dm_square_32"]["bf"],
        "compute_matrix_multiplication": True,
        "L": param.BASELINE * param.CONV_KM_TO_INV_EV,
        "ME": True,
        "MO": "NO",
    } | osc_param
    oscillated = pd.DataFrame()
    for key in ["minE", "maxE"]:
        oscillated[key] = to_oscillate[key]

    for key in ["numu", "nue", "antinumu", "antinue", "nutau", "antinutau"]:
        oscillated[key] = 0.0

    for i in range(len(oscillated)):
        Pee, Pem, Pet, Pme, Pmm, Pmt, Pte, Ptm, Ptt = osc.probability_general(
            E=(oscillated.maxE.loc[i] + oscillated.minE.loc[i]) / 2, **osc_param
        )
        oscillated.loc[i, "numu"] = (
            to_oscillate.loc[i, "numu"] * Pmm + to_oscillate.loc[i, "nue"] * Pem
        )
        oscillated.loc[i, "nue"] = (
            to_oscillate.loc[i, "nue"] * Pee + to_oscillate.loc[i, "numu"] * Pme
        )
        oscillated.loc[i, "nutau"] = (
            to_oscillate.loc[i, "nue"] * Pet + to_oscillate.loc[i, "numu"] * Pmt
        )

        Pee, Pem, Pet, Pme, Pmm, Pmt, Pte, Ptm, Ptt = osc.probability_general_anti(
            E=(oscillated.maxE.loc[i] + oscillated.minE.loc[i]) / 2, **osc_param
        )
        oscillated.loc[i, "antinumu"] = (
            to_oscillate.loc[i, "antinumu"] * Pmm + to_oscillate.loc[i, "antinue"] * Pem
        )
        oscillated.loc[i, "antinue"] = (
            to_oscillate.loc[i, "antinue"] * Pee + to_oscillate.loc[i, "antinumu"] * Pme
        )
        oscillated.loc[i, "antinutau"] = (
            to_oscillate.loc[i, "antinue"] * Pet + to_oscillate.loc[i, "antinumu"] * Pmt
        )

    return oscillated


def envelope(
    to_oscillate: pd.DataFrame,
    params: dict[str, list[float]],
    gate: pd.DataFrame = None,
) -> tuple[pd.DataFrame, ...]:
    if params == {}:
        raise ValueError("Empty envelope.")

    sample = {}
    list_of_df = []

    for key in params:
        sample[key] = np.linspace(params[key][0], params[key][1], 100)

    for key in params:
        for i in range(len(sample[key])):
            oscillated = oscillate(to_oscillate=to_oscillate, **{key: sample[key][i]})
            if gate is not None:
                oscillated = convolve(input=oscillated, gate=gate)
            list_of_df.append(oscillated)

    keys = np.linspace(0.0, 100.0 * len(sample), 100 * len(sample))
    oscillate_max = pd.concat(list_of_df, keys=keys).groupby(level=1).max()
    oscillate_min = pd.concat(list_of_df, keys=keys).groupby(level=1).min()

    return oscillate_min, oscillate_max
