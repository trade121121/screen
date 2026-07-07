"""
CRYPTO BOTTOM SCREENER — GitHub Actions Version
================================================
- Datenquelle: Bybit (kein Geo-Block auf US-Runnern)
- Läuft täglich nach Daily-Close via GitHub Actions
- Ausgabe: docs/index.html (GitHub Pages Dashboard) + Telegram

Zweistufige Weinstein-Logik:
  Stufe 1 "Basis baut": >EMA34 (5-40 Tage), Slope up, Retest bestätigt,
                        aus echter Basis kommend, ≤3x Jahrestief, ≤25% über EMA
  Stufe 2 "Ausbruch":   zusätzlich 90d-Hoch mit ≥1.5x Volumen
"""

import requests
import time
import os
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BINANCE_DATA = "https://data-api.binance.vision"  # öffentlicher Binance-Spiegel, kein Geo-Block

# ── Parameter (hier justieren) ────────────────────────────
MIN_DAYS_ABOVE_EMA = 5
MAX_DAYS_ABOVE_EMA = 40
MAX_DIST_ABOVE_EMA_PCT = 25.0
MIN_EMA_SLOPE_PCT = 0.0
PULLBACK_LOOKBACK_DAYS = 15
PULLBACK_MAX_DIST_PCT = 3.0
MAX_DIST_FROM_YEAR_LOW_X = 3.0
BASE_LOOKBACK_DAYS = 75
BASE_MIN_PCT_BELOW_EMA = 0.5
BREAKOUT_VOL_RATIO_MIN = 1.5
MIN_HISTORY_DAYS = 120

# Telegram aus Umgebungsvariablen (GitHub Secrets)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


@dataclass
class ScreenResult:
    symbol: str
    stage: int
    close: float
    dist_ema_pct: float
    ema_slope_pct: float
    days_above_ema: int
    dist_year_low_x: float
    vol_ratio: float
    drawdown_pct: float
    spark: list  # letzte 60 Closes für Sparkline


# ─────────────────────────────────────────────
# Bybit Daten
# ─────────────────────────────────────────────

EXCLUDE_SUFFIX = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
EXCLUDE_BASES = {"USDC", "FDUSD", "TUSD", "DAI", "EUR", "GBP", "BUSD", "USDP", "AEUR", "XUSD"}

def get_all_usdt_symbols() -> list:
    r = requests.get(f"{BINANCE_DATA}/api/v3/exchangeInfo", timeout=30)
    r.raise_for_status()
    symbols = []
    for s in r.json()["symbols"]:
        if (s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
                and not s["symbol"].endswith(EXCLUDE_SUFFIX)
                and s["baseAsset"] not in EXCLUDE_BASES):
            symbols.append(s["symbol"])
    return symbols


def get_daily_candles(symbol: str, limit: int = 400) -> Optional[list]:
    try:
        r = requests.get(f"{BINANCE_DATA}/api/v3/klines",
                         params={"symbol": symbol, "interval": "1d", "limit": limit},
                         timeout=20)
        r.raise_for_status()
        raw = r.json()  # älteste zuerst
        candles = [{
            'close': float(k[4]), 'high': float(k[2]),
            'low': float(k[3]), 'volume': float(k[7]),  # Quote-Volumen (USDT)
        } for k in raw]
        if len(candles) > 1:
            candles = candles[:-1]  # laufende Kerze entfernen
        return candles
    except Exception:
        return None


def ema(values: list, period: int) -> list:
    if len(values) < period:
        return []
    result = [None] * (period - 1)
    seed = sum(values[:period]) / period
    result.append(seed)
    mult = 2 / (period + 1)
    for v in values[period:]:
        seed = (v - seed) * mult + seed
        result.append(seed)
    return result


# ─────────────────────────────────────────────
# Screening
# ─────────────────────────────────────────────

def screen_coin(symbol: str, candles: list) -> Optional[ScreenResult]:
    if len(candles) < MIN_HISTORY_DAYS:
        return None

    closes = [c['close'] for c in candles]
    lows = [c['low'] for c in candles]
    highs = [c['high'] for c in candles]
    vols = [c['volume'] for c in candles]

    ema34 = ema(closes, 34)
    if not ema34 or ema34[-1] is None:
        return None

    close, e_now = closes[-1], ema34[-1]
    if close <= e_now:
        return None

    # Tage über EMA (5-40)
    days_above = 0
    for i in range(len(closes) - 1, -1, -1):
        if ema34[i] is None or closes[i] <= ema34[i]:
            break
        days_above += 1
    if not (MIN_DAYS_ABOVE_EMA <= days_above <= MAX_DAYS_ABOVE_EMA):
        return None

    # Nicht weggelaufen
    dist_above = (close - e_now) / e_now * 100
    if dist_above > MAX_DIST_ABOVE_EMA_PCT:
        return None

    # EMA-Slope up
    if len(ema34) < 6 or ema34[-6] is None:
        return None
    slope_pct = (e_now - ema34[-6]) / ema34[-6] * 100
    if slope_pct <= MIN_EMA_SLOPE_PCT:
        return None

    # Retest bestätigt
    pullback_ok = False
    lookback = min(PULLBACK_LOOKBACK_DAYS, days_above)
    for i in range(len(candles) - lookback, len(candles)):
        if ema34[i] is None:
            continue
        dist = (lows[i] - ema34[i]) / ema34[i] * 100
        if -1.0 <= dist <= PULLBACK_MAX_DIST_PCT:
            pullback_ok = True
            break
    if not pullback_ok:
        return None

    # Basis-Filter: davor überwiegend unter EMA
    streak_start = len(closes) - days_above
    base_start = max(0, streak_start - BASE_LOOKBACK_DAYS)
    below, total = 0, 0
    for i in range(base_start, streak_start):
        if ema34[i] is None:
            continue
        total += 1
        if closes[i] < ema34[i]:
            below += 1
    if total < 20 or below / total < BASE_MIN_PCT_BELOW_EMA:
        return None

    # Bottom-Kontext
    year = candles[-365:] if len(candles) >= 365 else candles
    year_low = min(c['low'] for c in year)
    dist_low_x = close / year_low if year_low > 0 else 999
    if dist_low_x > MAX_DIST_FROM_YEAR_LOW_X:
        return None

    # Stufe 2?
    high_90d = max(highs[-90:-1]) if len(highs) >= 91 else max(highs[:-1])
    avg_vol20 = sum(vols[-21:-1]) / 20 if len(vols) >= 21 else 1
    vol_ratio = vols[-1] / avg_vol20 if avg_vol20 > 0 else 0
    stage = 2 if (close > high_90d and vol_ratio >= BREAKOUT_VOL_RATIO_MIN) else 1

    ath = max(highs)
    return ScreenResult(
        symbol=symbol, stage=stage, close=close,
        dist_ema_pct=round(dist_above, 1),
        ema_slope_pct=round(slope_pct, 1),
        days_above_ema=days_above,
        dist_year_low_x=round(dist_low_x, 2),
        vol_ratio=round(vol_ratio, 1),
        drawdown_pct=round((close - ath) / ath * 100, 0),
        spark=[round(c, 8) for c in closes[-60:]],
    )


# ─────────────────────────────────────────────
# HTML Dashboard
# ─────────────────────────────────────────────

def sparkline_svg(values: list, color: str = "#2563eb", w: int = 120, h: int = 32) -> str:
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    rng = hi - lo if hi > lo else 1
    pts = []
    for i, v in enumerate(values):
        x = i / (len(values) - 1) * w
        y = h - (v - lo) / rng * (h - 4) - 2
        pts.append(f"{x:.1f},{y:.1f}")
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<polyline points="{" ".join(pts)}" fill="none" '
            f'stroke="{color}" stroke-width="1.5"/></svg>')


def tv_link(symbol: str) -> str:
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol}"


def build_html(stage1: list, stage2: list, btc_context: str, scanned: int) -> str:
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    def row(r: ScreenResult, highlight: bool = False) -> str:
        coin = r.symbol.replace('USDT', '')
        bg = ' style="background:#f0fdf4"' if highlight else ''
        color = "#16a34a" if highlight else "#2563eb"
        return f"""<tr{bg}>
<td><a href="{tv_link(r.symbol)}" target="_blank"><b>{coin}</b></a></td>
<td>{sparkline_svg(r.spark, color)}</td>
<td>${r.close:.6g}</td>
<td>{r.days_above_ema}d</td>
<td>+{r.dist_ema_pct}%</td>
<td>+{r.ema_slope_pct}%</td>
<td>{r.dist_year_low_x}x</td>
<td>{r.drawdown_pct:.0f}%</td>
<td>{r.vol_ratio}x</td>
</tr>"""

    s2_rows = "".join(row(r, True) for r in stage2)
    s1_rows = "".join(row(r) for r in stage1)

    s2_section = f"""
<h2>🚀 Ausbrüche (Stufe 2)</h2>
<p class="sub">90-Tage-Hoch mit erhöhtem Volumen — das Signal ist da.</p>
<table>
<tr><th>Coin</th><th>60 Tage</th><th>Kurs</th><th>über EMA34</th><th>Abstand EMA</th><th>Slope 5d</th><th>vom Jahrestief</th><th>Drawdown</th><th>Vol-Ratio</th></tr>
{s2_rows}
</table>""" if stage2 else "<h2>🚀 Ausbrüche (Stufe 2)</h2><p class='sub'>Keine Ausbrüche heute.</p>"

    s1_section = f"""
<h2>👀 Basis baut (Stufe 1 — Watchlist)</h2>
<p class="sub">Über EMA34 mit bestätigtem Retest, EMA dreht hoch, aus echter Basis. Beobachten — Ausbruch abwarten.</p>
<table>
<tr><th>Coin</th><th>60 Tage</th><th>Kurs</th><th>über EMA34</th><th>Abstand EMA</th><th>Slope 5d</th><th>vom Jahrestief</th><th>Drawdown</th><th>Vol-Ratio</th></tr>
{s1_rows}
</table>""" if stage1 else "<h2>👀 Basis baut (Stufe 1)</h2><p class='sub'>Keine Kandidaten heute.</p>"

    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Crypto Bottom Screener</title>
<style>
body {{ font-family: -apple-system, 'Segoe UI', sans-serif; max-width: 1100px;
       margin: 0 auto; padding: 24px; color: #1a1a1a; background: #fafafa; }}
h1 {{ font-size: 26px; margin-bottom: 4px; }}
h2 {{ font-size: 19px; margin-top: 36px; border-bottom: 2px solid #e5e5e5; padding-bottom: 6px; }}
.meta {{ color: #666; font-size: 13px; }}
.context {{ background: #fff; border-left: 4px solid #2563eb; padding: 12px 16px;
            margin: 20px 0; font-size: 15px; border-radius: 0 6px 6px 0; }}
.sub {{ color: #666; font-size: 13px; margin-top: -4px; }}
table {{ width: 100%; border-collapse: collapse; background: #fff;
         border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
th {{ text-align: left; font-size: 12px; color: #666; text-transform: uppercase;
      padding: 10px 12px; background: #f5f5f5; }}
td {{ padding: 10px 12px; border-top: 1px solid #f0f0f0; font-size: 14px; }}
a {{ color: #2563eb; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.footer {{ margin-top: 40px; color: #999; font-size: 12px; }}
</style>
</head>
<body>
<h1>Crypto Bottom Screener</h1>
<p class="meta">Weinstein Stage 1 → 2 · {scanned} Binance-Paare gescannt · Stand: {now}</p>
<div class="context">{btc_context}</div>
{s2_section}
{s1_section}
<p class="footer">Setup: Close &gt; EMA34 (5–40 Tage, Retest bestätigt, max 25% drüber) ·
EMA-Slope steigend · aus Basis kommend (≥50% der 75 Vortage unter EMA) ·
≤3× Jahrestief. Stufe 2: + 90d-Hoch mit ≥1.5× Volumen.<br>
Coin-Klick öffnet TradingView. ⚠️ Kein Handelsauftrag — nur Monitoring.</p>
</body>
</html>"""


def get_btc_context() -> str:
    candles = get_daily_candles("BTCUSDT", 400)
    if not candles or len(candles) < 200:
        return "BTC-Kontext nicht verfügbar."
    closes = [c['close'] for c in candles]
    sma200 = sum(closes[-200:]) / 200
    close = closes[-1]
    pct = (close - sma200) / sma200 * 100
    if close > sma200:
        return (f"<b>Bitcoin über der 200-Tage-Linie</b> ({pct:+.1f}%) — "
                f"konstruktives Umfeld für Bottom-Setups.")
    return (f"<b>Bitcoin unter der 200-Tage-Linie</b> ({pct:+.1f}%) — "
            f"Bärenmarkt-Kontext. Frühe Setups mit erhöhtem Fehlsignal-Risiko.")


# ─────────────────────────────────────────────
# Telegram
# ─────────────────────────────────────────────

def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.info("Kein Telegram konfiguriert")
        return
    try:
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                          json={"chat_id": TELEGRAM_CHAT_ID, "text": message,
                                "parse_mode": "HTML"}, timeout=10)
        r.raise_for_status()
    except Exception as e:
        log.error(f"Telegram: {e}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    log.info("🔍 Screener startet (Binance Spot-Daten)...")
    symbols = get_all_usdt_symbols()
    log.info(f"   {len(symbols)} USDT-Paare")

    stage1, stage2 = [], []
    for i, sym in enumerate(symbols):
        candles = get_daily_candles(sym)
        if candles:
            r = screen_coin(sym, candles)
            if r:
                (stage2 if r.stage == 2 else stage1).append(r)
                log.info(f"   {'🚀' if r.stage == 2 else '👀'} {sym}")
        if (i + 1) % 50 == 0:
            log.info(f"   {i+1}/{len(symbols)}")
        time.sleep(0.12)

    stage1.sort(key=lambda r: r.ema_slope_pct, reverse=True)
    stage2.sort(key=lambda r: r.vol_ratio, reverse=True)

    btc_context = get_btc_context()

    # HTML schreiben
    os.makedirs("docs", exist_ok=True)
    html = build_html(stage1, stage2, btc_context, len(symbols))
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"✅ docs/index.html geschrieben ({len(stage2)} Stufe 2, {len(stage1)} Stufe 1)")

    # Telegram (kompakt, mit Link zur Website)
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    site_url = ""
    if repo:
        user, name = repo.split("/")
        site_url = f"\n\n🔗 https://{user}.github.io/{name}/"

    lines = []
    if stage2:
        lines.append("🚀 <b>AUSBRÜCHE:</b>")
        for r in stage2:
            lines.append(f"✅ {r.symbol.replace('USDT','')} — {r.vol_ratio}x Vol, 90d-Hoch")
    if stage1:
        lines.append(f"\n👀 <b>Watchlist ({len(stage1)}):</b> " +
                     ", ".join(r.symbol.replace('USDT','') for r in stage1[:12]))
    if not lines:
        lines.append("Keine Kandidaten heute.")

    send_telegram(f"🔍 <b>BOTTOM SCREENER</b> — {datetime.now(timezone.utc).strftime('%d.%m.')}\n"
                  + "\n".join(lines) + site_url)
    log.info("Fertig.")


if __name__ == "__main__":
    main()
