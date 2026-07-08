"""
Project 6 — Crypto Funding-Carry
================================
Economic hypothesis (stated FIRST): the perpetual-futures funding rate is the
price of leverage. When funding is high, longs are crowded and pay shorts; the
crowded side tends to under-perform (mean-reversion of positioning), so FADING
extreme funding — and collecting the funding while you do — is a carry premium.

We test (1) whether funding predicts forward perp returns, and (2) whether a
funding-fade carry strategy survives net of cost. Uses a market many candidates
ignore (novel-data signal), on free Binance perp data.

Author: Christian Macion.
"""
import sys, os, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "quantlib"))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import quantlib as q

Z_WIN, ANN, COST_BPS = 9, 365*3, 5.0    # 9x8h = 3-day funding z; 3 funding stamps/day; 5bps round-turn

def main():
    print("Pulling BTC perp funding + perp 8h price (Binance) ...")
    fund = q.binance_funding("BTCUSDT", start="2019-09-01")["funding"]
    px = q.binance_perp_klines("BTCUSDT", "8h", start="2019-09-01")["close"]
    df = pd.DataFrame({"funding": fund}).join(px.rename("px"), how="inner").dropna()
    df["ret"] = df["px"].pct_change()
    df = df.dropna()
    print(f"  {len(df)} 8h periods  {df.index[0].date()}..{df.index[-1].date()}  "
          f"(avg funding {df['funding'].mean()*100*3*365:.1f}% annualized)")

    # (1) predictive: does funding predict the NEXT-period return? (expect negative)
    fz = (df["funding"] - df["funding"].rolling(Z_WIN).mean()) / df["funding"].rolling(Z_WIN).std()
    fwd = df["ret"].shift(-1)
    reg = pd.DataFrame({"fz": fz, "fwd": fwd}).dropna()
    pred_corr = reg["fz"].corr(reg["fwd"])
    print(f"\n(1) corr(funding z, next-period return) = {pred_corr:+.3f}  "
          f"({'crowded longs under-perform (thesis-consistent)' if pred_corr<0 else 'no reversion'})")

    # (2) carry strategy: fade funding extreme; period return = pos*(price_ret - funding)
    pos = -np.sign(fz).shift(1).fillna(0.0)                 # short when funding high, no look-ahead
    turn = pos.diff().abs().fillna(0.0)
    strat = (pos*(df["ret"] - df["funding"]) - turn*COST_BPS*1e-4).dropna()
    n=len(strat); split=int(n*0.6)
    is_s, oos_s = strat.iloc[:split], strat.iloc[split:]
    full,is_sh,oos_sh = q.sharpe(strat,ANN), q.sharpe(is_s,ANN), q.sharpe(oos_s,ANN)
    t=q.tstat(strat); lo,hi=q.block_bootstrap_ci(strat.values, block=30, reps=1500, ppy=ANN)
    print(f"\n(2) Funding-fade carry (net {COST_BPS}bps):")
    print(f"    Full Sharpe {full:.2f} | t {t:.2f} | IS {is_sh:.2f} | OOS {oos_sh:.2f}")
    print(f"    Sharpe block-CI [{lo:.2f},{hi:.2f}] ({'excludes 0' if (lo>0 or hi<0) else 'straddles 0'})")
    # era stability by year
    yr = strat.groupby(strat.index.year).apply(lambda s: q.sharpe(s,ANN))
    print(f"    Per-year Sharpe: " + ", ".join(f"{y}:{v:+.2f}" for y,v in yr.items()))

    fig, ax = plt.subplots(1,2, figsize=(11,4.2))
    ax[0].plot((1+strat).cumprod(), color="#1f6f6f", lw=1.5)
    ax[0].axvline(strat.index[split], color="crimson", ls="--", lw=1.1, label="IS/OOS")
    ax[0].set_title(f"Funding-carry equity (Sharpe {full:.2f}, IS {is_sh:.2f}/OOS {oos_sh:.2f})")
    ax[0].set_ylabel("growth of $1"); ax[0].legend(fontsize=8)
    ax[1].bar([str(y) for y in yr.index], yr.values, color="#1f6f6f")
    ax[1].axhline(0, color="gray", lw=0.7); ax[1].set_title("Per-year Sharpe (decay check)")
    plt.setp(ax[1].get_xticklabels(), rotation=45, fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(__file__),"results","figure.png"), dpi=130)

    json.dump({"n_periods":int(n),"pred_corr":round(float(pred_corr),4),
               "full_sharpe":round(full,3),"tstat":round(t,3),"is_sharpe":round(is_sh,3),
               "oos_sharpe":round(oos_sh,3),"boot_ci":[round(lo,3),round(hi,3)],
               "avg_funding_annualized_pct":round(float(df['funding'].mean()*100*3*365),2)},
              open(os.path.join(os.path.dirname(__file__),"results","results.json"),"w"), indent=2)
    print("\nSaved results/figure.png and results/results.json")

if __name__ == "__main__":
    main()
