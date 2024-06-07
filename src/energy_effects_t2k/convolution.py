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
import matplotlib.pyplot as plt


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


def bias(
    input: pd.DataFrame,
    a: float = 0.0,
    b: float = 0.0,
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

    for key in y_keys:
        interpolated = np.interp(
            x=(input.minE + input.maxE) / 2,
            xp=(1 + a) * (input.minE + input.maxE) / 2 + b,
            fp=input[key],
        )
        output[key] = interpolated

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


"""
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
"""


def envelope_list(
    to_oscillate: pd.DataFrame,
    vparams: dict[str, list[float]],
    fparams: dict[str, float] = {},
    gate: pd.DataFrame = None,
    n: int = 100,
) -> list[pd.DataFrame]:

    if len(vparams) == 0:
        if len(fparams) == 0:
            raise ValueError("Empty envelope.")

        else:
            oscillated = oscillate(to_oscillate=to_oscillate, **fparams)
            if gate is not None:
                oscillated = convolve(input=oscillated, gate=gate)
            return [oscillated]

    else:
        key, values = vparams.popitem()
        sample = np.linspace(*values, n)
        list_of_df = []

        for value in sample:
            list_of_df += envelope_list(
                to_oscillate=to_oscillate,
                vparams=vparams.copy(),
                fparams=fparams | {key: value},
                gate=gate,
                n=n,
            )

        return list_of_df


def envelope(
    to_oscillate: pd.DataFrame,
    params: dict[str, list[float]],
    gate: pd.DataFrame = None,
    n: int = 10,
) -> tuple[pd.DataFrame, ...]:

    m = len(params)
    keys = np.linspace(0.0, n**m, n**m)
    list_of_df = envelope_list(
        to_oscillate=to_oscillate, vparams=params.copy(), gate=gate, n=n
    )
    oscillate_max = pd.concat(list_of_df, keys=keys).groupby(level=1).max()
    oscillate_min = pd.concat(list_of_df, keys=keys).groupby(level=1).min()

    return oscillate_min, oscillate_max


carabadjac_transform = {
    "theta_12": ["theta_12", r"$\theta_{12}$"],
    "theta_13": ["theta_13", r"$\theta_{13}$"],
    "theta_23": ["theta_23", r"$\theta_{23}$"],
    "dm2_21": ["Dm_square_21", r"$\Delta m_{21}^2$"],
    "dm2_atm": ["Dm_square_32", r"$\Delta m_{32}^2$"],
    "delta_cp": ["delta_CP", r"$\delta_\text{CP}$"],
}

param_transform = {"bf": "BF", "p1sigma": r"$\pm 1\sigma$", "p3sigma": r"$\pm 3\sigma$"}

"""
def oscillation_analysis(
    number,
    energy_resolution,
    params,
    flavors=['numu', 'nue', 'anitnumu', 'antinue'],
):
    for key in ['bf', 'p1sigma', 'm1sigma', 'p3sigma', 'm3sigma']:
        number_oscillated[key] = oscillate(to_oscillate=number, **{params})
"""


def oscillated_plot(
    number_oscillated,
    oscillated_smeared,
    params,
    channel="",
    resolution="",
    keys=["numu"],
    number=None,
):
    fig, axs = plt.subplots(
        2,
        1,
        sharex="col",
        sharey="row",
        figsize=(15, 10),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    fig.subplots_adjust(wspace=0, hspace=0)

    to_print = ", ".join([carabadjac_transform[x][1] for x in params.keys()])

    for key in keys:
        axs[0].plot(
            (number_oscillated["bf"].minE + number_oscillated["bf"].maxE) / 2,
            number_oscillated["bf"][key],
            label=print_dict[key] + r" ($\times \mathcal{P}$) (" + to_print + " BF)",
            color="indianred",
        )

        axs[0].plot(
            (number_oscillated["p1sigma"].minE + number_oscillated["p1sigma"].maxE) / 2,
            number_oscillated["p1sigma"][key],
            label=print_dict[key]
            + r" ($\times \mathcal{P}$) ("
            + to_print
            + r" $\pm 1\sigma$)",
            color="indianred",
            linestyle="--",
        )

        axs[0].plot(
            (number_oscillated["m1sigma"].minE + number_oscillated["m1sigma"].maxE) / 2,
            number_oscillated["m1sigma"][key],
            color="indianred",
            linestyle="--",
        )

        axs[0].plot(
            (number_oscillated["p3sigma"].minE + number_oscillated["p3sigma"].maxE) / 2,
            number_oscillated["p3sigma"][key],
            label=print_dict[key]
            + r" ($\times \mathcal{P}$) ("
            + to_print
            + r" $\pm 3\sigma$)",
            color="indianred",
            linestyle="dotted",
        )

        axs[0].plot(
            (number_oscillated["m3sigma"].minE + number_oscillated["m3sigma"].maxE) / 2,
            number_oscillated["m3sigma"][key],
            color="indianred",
            linestyle="dotted",
        )

        axs[0].fill_between(
            (number_oscillated["max"].minE + number_oscillated["min"].maxE) / 2,
            number_oscillated["min"][key],
            number_oscillated["max"][key],
            # label=print_dict[key] + r" ($\times \mathcal{P})\ (\Delta m_{32}^2 \pm 3\sigma$)",
            color="indianred",
            alpha=0.5,
        )

        axs[0].plot(
            (oscillated_smeared["bf"].minE + oscillated_smeared["bf"].maxE) / 2,
            oscillated_smeared["bf"][key],
            label=print_dict[key]
            + r" ($\times \mathcal{P} \otimes \delta E$) ("
            + to_print
            + " BF)",
            color="cadetblue",
        )

        axs[0].plot(
            (oscillated_smeared["p1sigma"].minE + oscillated_smeared["p1sigma"].maxE)
            / 2,
            oscillated_smeared["p1sigma"][key],
            label=print_dict[key]
            + r" ($\times \mathcal{P} \otimes \delta E$) ("
            + to_print
            + r" $\pm 1\sigma$)",
            color="cadetblue",
            linestyle="--",
        )

        axs[0].plot(
            (oscillated_smeared["m1sigma"].minE + oscillated_smeared["m1sigma"].maxE)
            / 2,
            oscillated_smeared["m1sigma"][key],
            color="cadetblue",
            linestyle="--",
        )

        axs[0].plot(
            (oscillated_smeared["p3sigma"].minE + oscillated_smeared["p3sigma"].maxE)
            / 2,
            oscillated_smeared["p3sigma"][key],
            label=print_dict[key]
            + r" ($\times \mathcal{P} \otimes \delta E$) ("
            + to_print
            + r" $\pm 3\sigma$)",
            color="cadetblue",
            linestyle="dotted",
        )

        axs[0].plot(
            (oscillated_smeared["m3sigma"].minE + oscillated_smeared["m3sigma"].maxE)
            / 2,
            oscillated_smeared["m3sigma"][key],
            color="cadetblue",
            linestyle="dotted",
        )

        axs[0].fill_between(
            (oscillated_smeared["max"].minE + oscillated_smeared["min"].maxE) / 2,
            oscillated_smeared["max"][key],
            oscillated_smeared["min"][key],
            # label=print_dict[key] + r" ($\times \mathcal{P})\ (\Delta m_{32}^2 \pm 3\sigma$)",
            color="cadetblue",
            alpha=0.5,
        )

        axs[1].plot(
            (number_oscillated["bf"].minE + number_oscillated["bf"].maxE) / 2,
            number_oscillated["bf"][key] / number_oscillated["bf"][key],
            color="indianred",
        )

        axs[1].plot(
            (number_oscillated["p1sigma"].minE + number_oscillated["p1sigma"].maxE) / 2,
            number_oscillated["p1sigma"][key] / number_oscillated["bf"][key],
            color="indianred",
            linestyle="--",
        )

        axs[1].plot(
            (number_oscillated["m1sigma"].minE + number_oscillated["m1sigma"].maxE) / 2,
            number_oscillated["m1sigma"][key] / number_oscillated["bf"][key],
            color="indianred",
            linestyle="--",
        )

        axs[1].plot(
            (number_oscillated["p3sigma"].minE + number_oscillated["p3sigma"].maxE) / 2,
            number_oscillated["p3sigma"][key] / number_oscillated["bf"][key],
            color="indianred",
            linestyle="dotted",
        )

        axs[1].plot(
            (number_oscillated["m3sigma"].minE + number_oscillated["m3sigma"].maxE) / 2,
            number_oscillated["m3sigma"][key] / number_oscillated["bf"][key],
            color="indianred",
            linestyle="dotted",
        )

        axs[1].fill_between(
            (number_oscillated["max"].minE + number_oscillated["min"].maxE) / 2,
            number_oscillated["max"][key] / number_oscillated["bf"][key],
            number_oscillated["min"][key] / number_oscillated["bf"][key],
            color="indianred",
            alpha=0.5,
        )

        axs[1].plot(
            (oscillated_smeared["bf"].minE + oscillated_smeared["bf"].maxE) / 2,
            oscillated_smeared["bf"][key] / oscillated_smeared["bf"][key],
            color="cadetblue",
        )

        axs[1].plot(
            (oscillated_smeared["p1sigma"].minE + oscillated_smeared["p1sigma"].maxE)
            / 2,
            oscillated_smeared["p1sigma"][key] / oscillated_smeared["bf"][key],
            color="cadetblue",
            linestyle="--",
        )

        axs[1].plot(
            (oscillated_smeared["m1sigma"].minE + oscillated_smeared["m1sigma"].maxE)
            / 2,
            oscillated_smeared["m1sigma"][key] / oscillated_smeared["bf"][key],
            color="cadetblue",
            linestyle="--",
        )

        axs[1].plot(
            (oscillated_smeared["p3sigma"].minE + oscillated_smeared["p3sigma"].maxE)
            / 2,
            oscillated_smeared["p3sigma"][key] / oscillated_smeared["bf"][key],
            color="cadetblue",
            linestyle="dotted",
        )

        axs[1].plot(
            (oscillated_smeared["m3sigma"].minE + oscillated_smeared["m3sigma"].maxE)
            / 2,
            oscillated_smeared["m3sigma"][key] / oscillated_smeared["bf"][key],
            color="cadetblue",
            linestyle="dotted",
        )

        axs[1].fill_between(
            (oscillated_smeared["max"].minE + oscillated_smeared["min"].maxE) / 2,
            oscillated_smeared["max"][key] / oscillated_smeared["bf"][key],
            oscillated_smeared["min"][key] / oscillated_smeared["bf"][key],
            color="cadetblue",
            alpha=0.5,
        )

        if number is not None:
            axs[0].fill_between(
                (number.minE + number.maxE) / 2,
                number_oscillated["bf"][key]
                * (1 + 1 / np.sqrt(number[key] * (number.maxE - number.minE) * 1e3)),
                [
                    max(x, 0)
                    for x in number_oscillated["bf"][key]
                    * (1 - 1 / np.sqrt(number[key] * (number.maxE - number.minE) * 1e3))
                ],
                facecolor="none",
                edgecolor="indianred",
                hatch="/",
                alpha=0.5,
                zorder=-1,
                label=print_dict[key] + r" ($\times \mathcal{P}$) stat. uncertainty",
            )

            ylim = axs[1].get_ylim()

            axs[1].fill_between(
                (number.minE + number.maxE) / 2,
                [
                    min(1 + x, ylim[1])
                    for x in (
                        1 / (np.sqrt(number[key] * (number.maxE - number.minE) * 1e3))
                    ).replace([np.inf, -np.inf], [*ylim])
                ],
                [
                    max(1 - x, ylim[0])
                    for x in (
                        1 / (np.sqrt(number[key] * (number.maxE - number.minE) * 1e3))
                    ).replace([np.inf, -np.inf], [*ylim])
                ],
                facecolor="none",
                edgecolor="indianred",
                hatch="/",
                alpha=0.5,
                zorder=-1,
            )

            axs[1].set_ylim(*ylim)

    for ax in axs:
        ax.axvline(x=0.6, linestyle="--", color="black")
    axs[0].text(
        x=0.6,
        y=0.9,
        s=r"600 MeV",
        ha="right",
        va="top",
        rotation=90,
        transform=axs[0].get_xaxis_transform(),
    )

    if channel != "":
        props = dict(boxstyle="round", facecolor="yellow", alpha=0.5)
        axs[0].text(
            0.07,
            0.93,
            channel,
            transform=axs[0].transAxes,
            fontsize=20,
            va="center",
            ha="center",
            bbox=props,
        )

    if resolution != "":
        props = dict(boxstyle="round", facecolor="yellow", alpha=0.5)
        axs[0].text(
            0.07,
            0.88,
            r"$\delta E =$" + resolution,
            transform=axs[0].transAxes,
            fontsize=10,
            va="top",
            ha="center",
            bbox=props,
            # rotation=90
        )

    # plt.xscale("log")
    # plt.yscale("log")
    axs[1].set_xlabel(r"$E$ [GeV]")
    axs[0].set_ylabel(r"Number per energy bin [$\text{MeV}^{-1}$]")
    axs[1].set_ylabel(r"Normalized to BF")
    # plt.ylim(bottom=1e-1)
    axs[0].set_xlim([0.0, 1.4])
    # axs[1].set_ylim([0.9,1.12])
    # plt.ylim([0.02,0.1])
    if "anti" in keys[0]:
        plt.suptitle(
            r"Number of expected CCQE "
            + channel
            + r" at SK in (RHC | Nominal) with NuFIT24 param. in (NO | ME)"
        )
    else:
        plt.suptitle(
            r"Number of expected CCQE "
            + channel
            + r" at SK in (FHC | Nominal) with NuFIT24 param. in (NO | ME)"
        )

    if "nue" in keys[0]:
        for ax in axs:
            ax.fill_betweenx(
                y=[*ax.get_ylim()],
                x1=0.00773,
                color="black",
                alpha=0.2,
                label=r"Cherenkov detection limit",
            )
            ax.set_ymargin(0)

    if "numu" in keys[0]:
        for ax in axs:
            ax.fill_betweenx(
                y=[*ax.get_ylim()],
                x1=0.16,
                color="black",
                alpha=0.2,
                label=r"Cherenkov detection limit",
            )
            ax.set_ymargin(0)

    axs[0].legend()

    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()
