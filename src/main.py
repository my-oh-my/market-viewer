"""Main entry point for the application."""

import argparse
from src.processor import process_symbol


W20_SYMBOLS = [
    "ALE.WA",
    "ALR.WA",
    "BDX.WA",
    "CDR.WA",
    "CPS.WA",
    "DNP.WA",
    "JSW.WA",
    "KGH.WA",
    "KRU.WA",
    "KTY.WA",
    "LPP.WA",
    "MBK.WA",
    "OPL.WA",
    "PCO.WA",
    "PEO.WA",
    "PGE.WA",
    "PKN.WA",
    "PKO.WA",
    "PZU.WA",
    "SPL.WA",
]

MWIG40_SYMBOLS = [
    "11B.WA",
    "ABE.WA",
    "ACP.WA",
    "APR.WA",
    "ASB.WA",
    "ATC.WA",
    "ATT.WA",
    "BFT.WA",
    "BHW.WA",
    "BNP.WA",
    "CAR.WA",
    "CBF.WA",
    "CCC.WA",
    "CIG.WA",
    "CLN.WA",
    "DOM.WA",
    "DVL.WA",
    "EAT.WA",
    "ENA.WA",
    "EUR.WA",
    "GEA.WA",
    "GPP.WA",
    "GPW.WA",
    "HUG.WA",
    "ING.WA",
    "LWB.WA",
    "MBR.WA",
    "MIL.WA",
    "MRB.WA",
    "NEU.WA",
    "RBW.WA",
    "RVU.WA",
    "SLV.WA",
    "SNT.WA",
    "TEN.WA",
    "TPE.WA",
    "TXT.WA",
    "WPL.WA",
    "XTB.WA",
]

SWIG80_SYMBOLS = [
    "1AT.WA",
    "ABS.WA",
    "AGO.WA",
    "ALL.WA",
    "AMB.WA",
    "AMC.WA",
    "APT.WA",
    "ARH.WA",
    "ASE.WA",
    "AST.WA",
    "BCX.WA",
    "BIO.WA",
    "BLO.WA",
    "BMC.WA",
    "BOS.WA",
    "BRS.WA",
    "CLC.WA",
    "CMP.WA",
    "COG.WA",
    "CRI.WA",
    "CRJ.WA",
    "CTX.WA",
    "DAT.WA",
    "DCR.WA",
    "ECH.WA",
    "ELT.WA",
    "ENT.WA",
    "ERB.WA",
    "FRO.WA",
    "FTE.WA",
    "GRX.WA",
    "INK.WA",
    "KGN.WA",
    "LBW.WA",
    "MAB.WA",
    "MCI.WA",
    "MDG.WA",
    "MGT.WA",
    "MLG.WA",
    "MLS.WA",
    "MNC.WA",
    "MOC.WA",
    "MRC.WA",
    "MSZ.WA",
    "MUR.WA",
    "NWG.WA",
    "OND.WA",
    "OPN.WA",
    "PBX.WA",
    "PCE.WA",
    "PCF.WA",
    "PCR.WA",
    "PEN.WA",
    "PEP.WA",
    "PLW.WA",
    "PUR.WA",
    "PXM.WA",
    "SCP.WA",
    "SEL.WA",
    "SGN.WA",
    "SHO.WA",
    "SKA.WA",
    "SNK.WA",
    "STP.WA",
    "STX.WA",
    "SVE.WA",
    "TAR.WA",
    "TOA.WA",
    "TOR.WA",
    "UNT.WA",
    "VGO.WA",
    "VOT.WA",
    "VOX.WA",
    "VRC.WA",
    "VRG.WA",
    "WLT.WA",
    "WTN.WA",
    "WWL.WA",
    "XTP.WA",
    "ZEP.WA",
]


SYMBOLS = ",".join(W20_SYMBOLS + MWIG40_SYMBOLS + SWIG80_SYMBOLS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Viewer")
    parser.add_argument(
        "--symbols",
        type=str,
        default=SYMBOLS,
        help="Comma-separated list of symbols to fetch data for.",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="1d",
        help="Time period to fetch data for (e.g., '1d', '1mo', '1y').",
    )
    parser.add_argument(
        "--intervals",
        type=str,
        default="1h",
        help="Comma-separated list of intervals (e.g., '1m,5m,1h').",
    )
    parser.add_argument(
        "--plot-all",
        action="store_true",
        help="Plot a chart for every symbol, regardless of conditions.",
    )
    parser.add_argument(
        "--k-window",
        type=int,
        default=14,
        help="The window size for the %%K line in the Stochastic Oscillator.",
    )
    parser.add_argument(
        "--d-window",
        type=int,
        default=3,
        help="The window size for the %%D line (smoothing) in the Stochastic Oscillator.",
    )
    parser.add_argument(
        "--save-html-dir",
        type=str,
        default=None,
        help="Save the chart as an HTML file in the specified directory.",
    )

    # New Analysis Arguments
    parser.add_argument(
        "--volume-profile",
        action="store_true",
        help="Calculate and display Volume Profile.",
    )
    parser.add_argument(
        "--vp-bins",
        type=int,
        default=50,
        help="Number of bins for Volume Profile calculation.",
    )
    parser.add_argument(
        "--consolidation",
        action="store_true",
        help="Detect and highlight consolidation zones.",
    )
    parser.add_argument(
        "--consolidation-window",
        type=int,
        default=20,
        help="Window size for consolidation detection.",
    )
    parser.add_argument(
        "--consolidation-atr-multiplier",
        type=float,
        default=1.5,
        help="ATR multiplier for consolidation detection (default: 1.5).",
    )
    parser.add_argument(
        "--vwap",
        action="store_true",
        help="Calculate and display VWAP with Standard Deviation bands.",
    )
    parser.add_argument(
        "--support-resistance",
        action="store_true",
        help="Automatically detect and display Support and Resistance levels.",
    )
    parser.add_argument(
        "--ror",
        action="store_true",
        help="Calculate and display Rate of Return.",
    )

    args = parser.parse_args()

    symbols_list = [s.strip() for s in args.symbols.split(",")]
    for symbol in symbols_list:
        process_symbol(symbol, args)
