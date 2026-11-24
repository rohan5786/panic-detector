# Market panic detector: rolling correlation structure of equity returns.

import argparse

import numpy as np
import pandas as pd
import yfinance as yf

# current S&P 500 membership
UNIVERSE = {
    "big_tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
                 "META", "AVGO", "ORCL", "AMD", "CRM"],
    "defense_gov": ["LMT", "RTX", "NOC", "GD", "LHX",
                    "BA", "HII", "LDOS", "TDG", "TXT"],
    "finance": ["JPM", "BAC", "WFC", "C", "GS",
                "MS", "BLK", "SCHW", "AXP", "USB"],
}

UNIVERSE = {
    "big_tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
                 "META", "AVGO", "ORCL", "AMD", "CRM"],
    "defense_gov": ["LMT", "RTX", "NOC", "GD", "LHX",
                    "BA", "HII", "LDOS", "TDG", "TXT"],
    "finance": ["JPM", "BAC", "WFC", "C", "GS",
                "MS", "BLK", "SCHW", "AXP", "USB"],
}
SECTOR_OF = {t: s for s, ts in UNIVERSE.items() for t in ts}
 
 
def pick_tickers(sectors):
    """Flatten the chosen sectors into one ticker list."""
    return [t for s in sectors for t in UNIVERSE[s]]
 
 
# data
def load_returns(sectors, start, end):
    """Daily log returns, one column per ticker (T rows x n columns)."""
    import yfinance as yf  # imported here so --selftest works without it
 
    tickers = pick_tickers(sectors)
    prices = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    # drop tickers with lots of missing history (e.g. recent listings)
    prices = prices.dropna(axis=1, thresh=int(0.95 * len(prices)))
    return np.log(prices).diff().dropna()
 

# matematica
def avg_offdiag(C) -> float:
    C = np.asarray(C)
    n = C.shape[0]
    return (np.sum(C) - n) / (n * (n - 1))
 
# eigvalsh returns ascending so get last index for largest
def absorption_ratio(C) -> float:
    C = np.asarray(C)
    n = C.shape[0]
    return np.linalg.eigvalsh(C)[-1] / n
 
 
def rolling_stats(returns, window):
    rows, dates = [], []
    for t in range(window, len(returns) + 1):
        C = returns.iloc[t - window:t].corr().to_numpy()
        rows.append((avg_offdiag(C), absorption_ratio(C)))
        dates.append(returns.index[t - 1])
    return pd.DataFrame(rows, index=dates, columns=["avg_corr", "absorption"])
 

# calm, elevated, panic
def classify_regime(absorption_now, history):
    p75, p90 = np.percentile(history, [75, 90])
    if absorption_now >= p90:
        return "PANIC"
    if absorption_now >= p75:
        return "ELEVATED"
    return "CALM"
 
 
# radar chart
def stock_angles(tickers):
    """Fixed angle (radians) per ticker, one wedge per sector present.
 
    Angles depend only on the ticker set, never on the date, so the picture
    cannot jitter from frame to frame.
    """
    sectors = [s for s in UNIVERSE if any(SECTOR_OF[t] == s for t in tickers)]
    wedge = 2 * np.pi / len(sectors)
    angles = {}
    for k, s in enumerate(sectors):
        members = [t for t in tickers if SECTOR_OF[t] == s]
        for j, t in enumerate(members):
            angles[t] = k * wedge + (j + 0.5) / len(members) * wedge
    return pd.Series(angles)
 
 
def radar_radii(window_returns):
    """Radius per stock: d = sqrt(2 * (1 - rho_i,m)), m = equal-weight market return."""
    m = window_returns.mean(axis=1)
    rho = window_returns.corrwith(m)
    return np.sqrt(np.clip(2 * (1 - rho), 0, None))
 
 
RINGS = [("CORE (d < 0.6)", 0.0, 0.6),
         ("MIDDLE (0.6 <= d < 0.9)", 0.6, 0.9),
         ("OUTER (d >= 0.9)", 0.9, np.inf)]
 
 
def render_text_radar(angles, radii, regime, date, stats_row):
    """Text radar: stocks bucketed into rings by radius, grouped by sector."""
    lines = [
        "\n\n"
        f"{date:%Y-%m-%d}   REGIME: {regime}",
        f"absorption (lambda1/n) = {stats_row['absorption']:.3f}   "
        f"avg correlation = {stats_row['avg_corr']:.3f}   "
        f"mean radius = {radii.mean():.3f}",
        "\n",
    ]
    for name, lo, hi in RINGS:
        members = radii[(radii >= lo) & (radii < hi)]
        lines.append(name)
        if members.empty:
            lines.append("    (none)")
            continue
        for sector in UNIVERSE:
            tick = [t for t in members.index if SECTOR_OF[t] == sector]
            if tick:
                tick.sort(key=lambda t: angles[t])
                cells = ", ".join(f"{t} {members[t]:.2f}" for t in tick)
                lines.append(f"    {sector:<12} {cells}")
    return "\n".join(lines)
 
 
# plots image
def plot_stats(stats, window, path="panic_stats.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
 
    # expanding percentiles use only the past; wait a year before trusting them
    p75 = stats["absorption"].expanding(min_periods=250).quantile(0.75)
    p90 = stats["absorption"].expanding(min_periods=250).quantile(0.90)
 
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    a.plot(stats.index, stats["avg_corr"], color="tab:blue")
    a.set_ylabel("avg pairwise correlation")
    a.set_title(f"Rolling correlation structure (window = {window} days)")
    b.plot(stats.index, stats["absorption"], color="tab:red", label="lambda1 / n")
    b.plot(p75.index, p75, "--", color="gray", lw=1, label="75th pct (past only)")
    b.plot(p90.index, p90, "--", color="black", lw=1, label="90th pct (past only)")
    b.set_ylabel("absorption ratio")
    b.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
 
 
def selftest():
    # uniform-rho matrix: closed forms from the derivation
    for n, rho in [(20, 0.2), (20, 0.8), (30, 0.05)]:
        C = np.full((n, n), rho)
        np.fill_diagonal(C, 1)
        assert np.isclose(avg_offdiag(C), rho)
        assert np.isclose(absorption_ratio(C), (1 + (n - 1) * rho) / n)
    print("uniform-rho tests passed")
 
    # synthetic returns with a calm regime then a one-factor "panic"
    rng = np.random.default_rng(0)
    T, n = 600, 20
    calm = rng.normal(size=(300, n))
    factor = rng.normal(size=(300, 1)) * 3
    panic = rng.normal(size=(300, n)) + factor
    X = pd.DataFrame(np.vstack([calm, panic]),
                     index=pd.bdate_range("2020-01-01", periods=T),
                     columns=pick_tickers(UNIVERSE)[:n])
    st = rolling_stats(X, 60)
    assert st["absorption"].iloc[-1] > 3 * st["absorption"].iloc[100]
    print("synthetic regime-change test passed")
    d = st.index[-1]
    reg = classify_regime(st["absorption"].iloc[-1], st["absorption"].loc[:d])
    print(render_text_radar(stock_angles(list(X.columns)),
                            radar_radii(X.iloc[-60:]), reg, d, st.iloc[-1]))
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sectors", nargs="+", choices=UNIVERSE.keys(),
                    default=list(UNIVERSE))
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-11-01")
    ap.add_argument("--window", type=int, default=120)
    ap.add_argument("--date", default=None, help="radar date, default = latest")
    # ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
 
    # if args.selftest:
    #     selftest()
    #     return
 
    returns = load_returns(args.sectors, args.start, args.end)
    n = returns.shape[1]
    print(f"{n} stocks, {len(returns)} days, window={args.window} "
          f"(T/n = {args.window / n:.1f})")
 
    stats = rolling_stats(returns, args.window)
    print("saved", plot_stats(stats, args.window))
 
    # pick the radar date: last trading day on or before --date
    if args.date is None:
        pos = len(returns)
    else:
        pos = returns.index.searchsorted(pd.Timestamp(args.date), side="right")
    if pos < args.window:
        raise SystemExit("not enough history before that date for one window")
    date = returns.index[pos - 1]
 
    history = stats["absorption"].loc[:date]          # past and present only
    regime = classify_regime(stats.loc[date, "absorption"], history)
    window_returns = returns.iloc[pos - args.window:pos]
    print(render_text_radar(stock_angles(list(returns.columns)),
                            radar_radii(window_returns), regime, date,
                            stats.loc[date]))
 
 
if __name__ == "__main__":
    main()