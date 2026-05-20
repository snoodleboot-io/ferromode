#!/usr/bin/env python3
"""
Generate IMF decomposition visualizations from S&P 500 data.

Reads individual stock CSVs from the downloaded Kaggle dataset, runs
ferromode EMD decomposition, and writes SVG charts to the output directory.

The output SVGs can be referenced directly in the Ferromode website
examples page (docs/examples.html).

Requirements:
    pip install ferromode numpy pandas matplotlib

Usage:
    # Single ticker
    python scripts/generate_imfs.py --ticker AAPL

    # Multiple tickers
    python scripts/generate_imfs.py --ticker AAPL MSFT NVDA

    # Custom paths
    python scripts/generate_imfs.py \\
        --ticker AAPL \\
        --data-dir data/sp500/individual_stocks_5yr \\
        --output docs/images \\
        --signal log-returns

    # Use all_stocks_5yr.csv instead of individual files
    python scripts/generate_imfs.py \\
        --merged-csv data/sp500/all_stocks_5yr.csv \\
        --ticker AAPL
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Colour palette matching the Ferromode website ───────────────────────────
PALETTE = {
    "iron":       "#0d0d0d",
    "iron_mid":   "#161616",
    "iron_light": "#1e1e1e",
    "border":     "#2a2826",
    "rust":       "#e0521a",
    "rust_hot":   "#ff6b2b",
    "patina":     "#3d9e96",
    "patina_hot": "#52c4bb",
    "bronze":     "#c4973d",
    "text":       "#f0ede8",
    "text_muted": "#9a9490",
    "text_dim":   "#5c5854",
}

# IMF line colours — cycle through these
IMF_COLORS = [
    PALETTE["rust"],
    PALETTE["patina"],
    PALETTE["bronze"],
    PALETTE["rust_hot"],
    PALETTE["patina_hot"],
    "#e8c87d",   # light bronze
    "#c47a52",   # muted rust
    "#6bbdb8",   # light patina
]


def load_matplotlib():
    """Import matplotlib with the non-interactive Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    return plt, mticker


def load_ferromode():
    """Import ferromode, with a helpful error if not installed."""
    try:
        import ferromode
        return ferromode
    except ImportError:
        print(
            "ERROR: ferromode not installed.\n"
            "  pip install ferromode\n"
            "  (or build from source: cargo build --release && pip install -e python/)"
        )
        sys.exit(1)


def load_ticker_data(ticker: str, data_dir: Path | None, merged_csv: Path | None) -> pd.Series:
    """Load adjusted close prices for a single ticker."""
    ticker_upper = ticker.upper()

    if merged_csv and merged_csv.exists():
        df = pd.read_csv(merged_csv, parse_dates=["date"])
        sub = df[df["Name"] == ticker_upper].copy()
        if sub.empty:
            raise ValueError(f"Ticker '{ticker_upper}' not found in {merged_csv}")
        sub = sub.sort_values("date").set_index("date")
        return sub["close"]

    if data_dir:
        # individual_stocks_5yr layout: TICKER_data.csv
        candidates = [
            data_dir / f"{ticker_upper}_data.csv",
            data_dir / f"{ticker_upper}.csv",
            data_dir / "individual_stocks_5yr" / f"{ticker_upper}_data.csv",
        ]
        for path in candidates:
            if path.exists():
                df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
                df = df.set_index("date")
                return df["close"]

    raise FileNotFoundError(
        f"Cannot locate data for '{ticker_upper}'.\n"
        f"  Searched: {data_dir}\n"
        f"  Expected: {ticker_upper}_data.csv or {ticker_upper}.csv\n"
        f"  Run: python scripts/download_sp500.py --unzip"
    )


def prepare_signal(prices: pd.Series, signal_type: str) -> tuple[np.ndarray, str, str]:
    """Convert prices to the requested signal type. Returns (array, ylabel, title_suffix)."""
    prices = prices.dropna()
    if signal_type == "log-returns":
        arr = np.diff(np.log(prices.values))
        return arr, "Log Return", "Log Returns"
    if signal_type == "returns":
        arr = prices.pct_change().dropna().values
        return arr, "Return", "Daily Returns"
    if signal_type == "price":
        arr = prices.values.astype(float)
        return arr, "Price (USD)", "Adjusted Close"
    raise ValueError(f"Unknown signal type '{signal_type}'. Choose: log-returns, returns, price")


def decompose(signal: np.ndarray, max_imf: int, tolerance: float, ferromode) -> tuple:
    """Run EMD decomposition."""
    emd = ferromode.EMD(
        max_imf=max_imf,
        max_sift=300,
        tolerance=tolerance,
        spline="cubic",
    )
    return emd.decompose(signal)


def plot_decomposition(
    ticker: str,
    signal: np.ndarray,
    imfs: list[np.ndarray],
    residual: np.ndarray,
    signal_label: str,
    title_suffix: str,
    output_path: Path,
) -> None:
    plt, mticker = load_matplotlib()

    n_rows = len(imfs) + 2  # original + IMFs + residual
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(12, 2.2 * n_rows),
        facecolor=PALETTE["iron"],
    )
    if n_rows == 1:
        axes = [axes]

    fig.suptitle(
        f"{ticker} — EMD Decomposition ({title_suffix})",
        color=PALETTE["text"],
        fontsize=13,
        fontweight="bold",
        y=1.002,
    )

    x = np.arange(len(signal))

    def style_ax(ax, title: str, color: str, y_label: str, last: bool = False) -> None:
        ax.set_facecolor(PALETTE["iron_mid"])
        ax.set_title(title, color=PALETTE["text_muted"], fontsize=8.5,
                     pad=4, loc="left", fontfamily="monospace")
        ax.set_ylabel(y_label, color=PALETTE["text_dim"], fontsize=7.5, labelpad=6)
        ax.tick_params(axis="both", colors=PALETTE["text_dim"], labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune="both"))
        if not last:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Sample index", color=PALETTE["text_dim"], fontsize=8)
        ax.tick_params(labelbottom=last)

    # ── Row 0: original signal ──────────────────────────────────────────────
    axes[0].plot(x, signal, color=PALETTE["rust"], linewidth=0.75, alpha=0.9)
    axes[0].axhline(0, color=PALETTE["border"], linewidth=0.5, linestyle="--")
    style_ax(axes[0], f"Original · {signal_label}", PALETTE["rust"], signal_label)

    # ── IMF rows ────────────────────────────────────────────────────────────
    for i, imf in enumerate(imfs):
        color = IMF_COLORS[i % len(IMF_COLORS)]
        xi = np.arange(len(imf))
        axes[i + 1].plot(xi, imf, color=color, linewidth=0.7, alpha=0.9)
        axes[i + 1].axhline(0, color=PALETTE["border"], linewidth=0.5, linestyle="--")

        # Variance explained
        var_pct = np.var(imf) / np.var(signal) * 100 if np.var(signal) > 0 else 0.0
        label = f"IMF {i+1}  ·  {var_pct:.1f}% var"
        style_ax(axes[i + 1], label, color, "Amplitude")

    # ── Residual row ────────────────────────────────────────────────────────
    xr = np.arange(len(residual))
    axes[-1].plot(xr, residual, color=PALETTE["patina"], linewidth=0.9, alpha=0.9)
    axes[-1].axhline(0, color=PALETTE["border"], linewidth=0.5, linestyle="--")
    style_ax(axes[-1], "Residual  ·  secular trend", PALETTE["patina"],
             "Amplitude", last=True)

    plt.tight_layout(h_pad=0.4)
    fig.patch.set_facecolor(PALETTE["iron"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="svg", bbox_inches="tight",
                facecolor=PALETTE["iron"], dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ticker", nargs="+", default=["AAPL"],
        metavar="TICKER",
        help="One or more ticker symbols (default: AAPL)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/sp500"),
        metavar="DIR",
        help="Directory containing individual stock CSVs (default: data/sp500)",
    )
    parser.add_argument(
        "--merged-csv", type=Path, default=None,
        metavar="CSV",
        help="Path to all_stocks_5yr.csv (overrides --data-dir lookup)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("webpage/images"),
        metavar="DIR",
        help="Output directory for SVG files (default: webpage/images)",
    )
    parser.add_argument(
        "--signal", choices=["log-returns", "returns", "price"], default="log-returns",
        help="Signal to decompose (default: log-returns)",
    )
    parser.add_argument(
        "--max-imf", type=int, default=6,
        help="Maximum number of IMFs (default: 6)",
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.005,
        help="EMD stopping criterion tolerance (default: 0.005)",
    )
    args = parser.parse_args()

    ferromode = load_ferromode()

    for ticker in args.ticker:
        print(f"\n[{ticker}]")
        try:
            prices = load_ticker_data(ticker, args.data_dir, args.merged_csv)
        except (FileNotFoundError, ValueError) as e:
            print(f"  SKIP: {e}")
            continue

        print(f"  Loaded {len(prices)} trading days  ({prices.index[0].date()} – {prices.index[-1].date()})")

        signal, ylabel, title_suffix = prepare_signal(prices, args.signal)
        print(f"  Signal: {args.signal}, {len(signal)} samples, "
              f"mean={np.mean(signal):.6f}, std={np.std(signal):.6f}")

        print(f"  Running EMD (max_imf={args.max_imf}, tol={args.tolerance}) …")
        try:
            imfs, residual = decompose(signal, args.max_imf, args.tolerance, ferromode)
        except Exception as e:
            print(f"  ERROR during decomposition: {e}")
            continue

        print(f"  Extracted {len(imfs)} IMFs + residual")

        # Reconstruction check
        reconstructed = sum(imfs) + residual
        error = np.max(np.abs(signal - reconstructed[:len(signal)]))
        print(f"  Reconstruction max error: {error:.2e}")

        output_path = args.output / f"{ticker.upper()}_imf_decomposition.svg"
        plot_decomposition(
            ticker=ticker.upper(),
            signal=signal,
            imfs=imfs,
            residual=residual,
            signal_label=ylabel,
            title_suffix=title_suffix,
            output_path=output_path,
        )

    print("\nAll done.")
    print(f"SVGs written to: {args.output}")
    print(
        "\nTo embed in the website, update the <object data=...> tags in webpage/examples.html\n"
        "to point at the generated SVG files."
    )


if __name__ == "__main__":
    main()
