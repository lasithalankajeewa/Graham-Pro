"""
Graham analysis engine.
calculate_full_analysis(data) → rich analysis dict
calculate_graham_score(data)  → backward-compat 5-tuple
"""


def _f(v):
    try:
        return float(v or 0)
    except Exception:
        return 0.0


def _pct(curr, prev):
    """Safe percentage change. Returns None when previous is zero/missing."""
    try:
        c, p = float(curr or 0), float(prev or 0)
        if p == 0:
            return None
        return (c - p) / abs(p) * 100
    except Exception:
        return None


def _fmt(v, decimals=2, suffix=""):
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}{suffix}"


def calculate_full_analysis(data: dict) -> dict:
    """
    Compute the complete Graham analysis from extracted PDF data.
    Returns a dict with all ratios, score table, MOS, checklist, and recommendation.
    Falls back to AI-extracted ratios when raw inputs are unavailable.
    """
    is_quarterly = data.get('report_type', '').lower() == 'quarterly'

    # ── Raw inputs ───────────────────────────────────────────────────────────
    eps              = _f(data.get('eps'))
    eps_prev         = _f(data.get('eps_prev'))
    revenue          = _f(data.get('revenue'))
    revenue_prev     = _f(data.get('revenue_prev'))
    net_income       = _f(data.get('net_income'))
    net_income_prev  = _f(data.get('net_income_prev'))

    total_equity        = _f(data.get('total_equity'))
    total_assets        = _f(data.get('total_assets'))
    total_liabilities   = _f(data.get('total_liabilities'))
    current_assets      = _f(data.get('current_assets'))
    current_liabilities = _f(data.get('current_liabilities'))

    operating_cf = _f(data.get('operating_cash_flow'))
    shares       = _f(data.get('shares_outstanding'))
    market_price = _f(data.get('market_price'))
    dps          = _f(data.get('dividend_per_share'))

    # AI-extracted fallback ratios (used when raw inputs are missing)
    ai_pe  = _f(data.get('pe_ratio'))
    ai_pb  = _f(data.get('pb_ratio'))
    ai_roe = _f(data.get('roe'))
    ai_de  = _f(data.get('debt_to_equity'))

    # Derive EPS from net_income / shares when AI extracted 0 but components exist.
    # Both fields are in millions (LKR millions / million shares = LKR per share).
    if eps == 0 and net_income > 0 and shares > 0:
        eps = net_income / shares
    if eps_prev == 0 and net_income_prev > 0 and shares > 0:
        eps_prev = net_income_prev / shares

    # ── Growth Rates ─────────────────────────────────────────────────────────
    revenue_growth = _pct(revenue, revenue_prev)
    profit_growth  = _pct(net_income, net_income_prev)
    eps_growth_yoy = _pct(eps, eps_prev)
    eps_growth = eps_growth_yoy if eps_growth_yoy is not None else (
        _f(data.get('earnings_growth_5yr')) or None
    )

    # ── Liabilities (derived if not directly given) ───────────────────────────
    liab = total_liabilities if total_liabilities > 0 else (
        (total_assets - total_equity) if total_assets > total_equity > 0 else 0
    )

    # ── Bank detection: financial institutions have >85% of assets funded by liabilities
    is_bank = (total_assets > 0) and (liab / total_assets > 0.85)

    # ── Quarterly annualization ───────────────────────────────────────────────
    # For quarterly reports we annualize EPS and profitability metrics (×4)
    # so P/E, MOS, ROE, and ROA are comparable to annual figures.
    eps_for_pe  = eps * 4 if is_quarterly else eps
    eps_for_iv  = eps * 4 if is_quarterly else eps

    # ── Valuation Ratios ─────────────────────────────────────────────────────
    pe   = (market_price / eps_for_pe) if (market_price > 0 and eps_for_pe > 0) else (ai_pe or None)

    bvps = (total_equity / shares) if (total_equity > 0 and shares > 0) else 0
    pb   = (market_price / bvps)   if (market_price > 0 and bvps > 0) else (ai_pb or None)

    cfps = (operating_cf / shares) if (operating_cf > 0 and shares > 0) else 0
    pcf  = (market_price / cfps)   if (market_price > 0 and cfps > 0) else None

    # ── Profitability Ratios ─────────────────────────────────────────────────
    roe_raw = (net_income / total_equity * 100) if (total_equity > 0 and net_income != 0) else (ai_roe or None)
    roa_raw = (net_income / total_assets * 100) if (total_assets > 0 and net_income != 0) else None
    # Annualize for quarterly
    roe = (roe_raw * 4) if (is_quarterly and roe_raw is not None) else roe_raw
    roa = (roa_raw * 4) if (is_quarterly and roa_raw is not None) else roa_raw

    de  = (liab / total_equity) if (total_equity > 0 and liab > 0) else (ai_de or None)
    current_ratio = (current_assets / current_liabilities) if current_liabilities > 0 else 0

    # ── Dividend ─────────────────────────────────────────────────────────────
    div_yield    = (dps / market_price * 100) if (market_price > 0 and dps > 0) else 0
    payout_ratio = (dps / eps * 100)          if (eps > 0 and dps > 0) else 0

    # ── Graham Score Table ────────────────────────────────────────────────────
    score = 0.0
    score_table = []
    checklist = []

    def _row(criterion, max_pts, value_str, earned):
        nonlocal score
        score += earned
        score_table.append({
            "Criterion": criterion,
            "Points": f"{earned}/{max_pts}",
            "Value": value_str,
            "Score": earned,
        })
        icon = "✅" if earned > 0 else "❌"
        checklist.append(f"{icon} {criterion}: {value_str} (+{earned})")

    if is_quarterly:
        # ── QUARTERLY SCORECARD ───────────────────────────────────────────────
        # P/E (annualized)
        pe_label = "P/E annualized < 15 (3pts) or 15–20 (1pt)"
        if pe and pe > 0:
            pts = 3 if pe < 15 else (1 if pe <= 20 else 0)
            _row(pe_label, 3, _fmt(pe, 1), pts)
        else:
            _row(pe_label, 3, "N/A", 0)

        # P/B
        pb_label = "P/B < 1.0 (3pts) or 1.0–1.5 (2pts)"
        if pb and pb > 0:
            pts = 3 if pb < 1.0 else (2 if pb <= 1.5 else 0)
            _row(pb_label, 3, _fmt(pb, 2), pts)
        else:
            _row(pb_label, 3, "N/A", 0)

        # EPS Growth YoY — extra 0.5pt tier for positive-but-small growth
        if eps_growth is not None:
            if eps_growth > 20:
                pts = 2
            elif eps_growth > 10:
                pts = 1
            elif eps_growth > 0:
                pts = 0.5
            else:
                pts = 0
            _row("YoY EPS Growth >20% (2pts), >10% (1pt), >0% (0.5pt)", 2, _fmt(eps_growth, 1, "%"), pts)
        else:
            _row("YoY EPS Growth >20% (2pts), >10% (1pt), >0% (0.5pt)", 2, "N/A", 0)

        # ROE — annualized
        roe_label = "Annualized ROE >15% (2pts) or >10% (1pt)"
        if roe is not None:
            pts = 2 if roe > 15 else (1 if roe > 10 else 0)
            _row(roe_label, 2, _fmt(roe, 1, "%"), pts)
        else:
            _row(roe_label, 2, "N/A", 0)

        # Non-bank only: Current Ratio and D/E
        if is_bank:
            # Banks: skip these criteria entirely — they are structurally inapplicable
            pass
        else:
            if current_ratio > 0:
                pts = 1 if current_ratio > 2.0 else 0
                _row("Current Ratio >2 (1pt) [non-banks]", 1, _fmt(current_ratio, 2), pts)
            else:
                _row("Current Ratio >2 (1pt) [non-banks]", 1, "N/A", 0)

            if de is not None and de >= 0:
                pts = 2 if de < 0.5 else 0
                _row("Debt/Equity <0.5 (2pts) [non-banks]", 2, _fmt(de, 2), pts)
            else:
                _row("Debt/Equity <0.5 (2pts) [non-banks]", 2, "N/A", 0)

        # Dividend Yield
        pts = 2 if div_yield > 5 else 0
        _row("Dividend Yield >5% (2pts)", 2, _fmt(div_yield, 1, "%") if div_yield > 0 else "0%", pts)

        # Grade thresholds — scale for banks (max 12) vs non-banks (max 15)
        max_score = 12 if is_bank else 15
        if max_score == 12:
            grade_thresholds = [(8, "A", "Strong Buy"), (6, "B", "Buy"), (4, "C", "Hold")]
        else:
            grade_thresholds = [(10, "A", "Strong Buy"), (7, "B", "Buy"), (4, "C", "Hold")]

        grade, rec = "D", "Sell"
        for threshold, g, r in grade_thresholds:
            if score >= threshold:
                grade, rec = g, r
                break

    else:
        # ── ANNUAL SCORECARD ──────────────────────────────────────────────────
        # P/E
        if pe and pe > 0:
            pts = 3 if pe < 15 else (1 if pe <= 20 else 0)
            _row("P/E < 15 (3pts) or 15–20 (1pt)", 3, _fmt(pe, 1), pts)
        else:
            _row("P/E < 15 (3pts) or 15–20 (1pt)", 3, "N/A", 0)

        # P/B
        if pb and pb > 0:
            pts = 3 if pb < 1.0 else (2 if pb <= 1.5 else 0)
            _row("P/B < 1.0 (3pts) or 1.0–1.5 (2pts)", 3, _fmt(pb, 2), pts)
        else:
            _row("P/B < 1.0 (3pts) or 1.0–1.5 (2pts)", 3, "N/A", 0)

        # EPS growth
        if eps_growth is not None:
            pts = 2 if eps_growth > 20 else (1 if eps_growth > 10 else 0)
            _row("EPS Growth >20% (2pts) or >10% (1pt)", 2, _fmt(eps_growth, 1, "%"), pts)
        else:
            _row("EPS Growth >20% (2pts) or >10% (1pt)", 2, "N/A", 0)

        # ROE
        if roe is not None:
            pts = 2 if roe > 15 else (1 if roe > 10 else 0)
            _row("ROE >15% (2pts) or >10% (1pt)", 2, _fmt(roe, 1, "%"), pts)
        else:
            _row("ROE >15% (2pts) or >10% (1pt)", 2, "N/A", 0)

        # D/E
        if de is not None and de >= 0:
            pts = 2 if de < 0.5 else (1 if de < 1.0 else 0)
            _row("Debt/Equity <0.5 (2pts) or <1.0 (1pt)", 2, _fmt(de, 2), pts)
        else:
            _row("Debt/Equity <0.5 (2pts) or <1.0 (1pt)", 2, "N/A", 0)

        # Current Ratio
        if current_ratio > 0:
            pts = 1 if current_ratio > 2.0 else 0
            _row("Current Ratio >2 (1pt)", 1, _fmt(current_ratio, 2), pts)
        else:
            _row("Current Ratio >2 (1pt)", 1, "N/A", 0)

        # Dividend Yield
        pts = 2 if div_yield > 5 else 0
        _row("Dividend Yield >5% (2pts)", 2, _fmt(div_yield, 1, "%") if div_yield > 0 else "0%", pts)

        if   score >= 10: grade, rec = "A", "Strong Buy"
        elif score >= 7:  grade, rec = "B", "Buy"
        elif score >= 4:  grade, rec = "C", "Hold"
        else:             grade, rec = "D", "Sell"

    # ── Margin of Safety (Sri Lanka formula) ─────────────────────────────────
    # IV = EPS × (8.5 + 2g) × (4.4/Y)  with g=5%, Y=11%  → IV = EPS × 7.4
    # For quarterly: uses annualized EPS so MOS is on the same basis as annual
    iv = eps_for_iv * 7.4 if eps_for_iv > 0 else 0
    mos_pct = ((iv - market_price) / iv * 100) if (iv > 0 and market_price > 0) else 0

    if   mos_pct >= 30: mos_grade = "Excellent (>30%)"
    elif mos_pct >= 20: mos_grade = "Good (20–30%)"
    elif mos_pct >= 10: mos_grade = "Moderate (10–20%)"
    elif mos_pct >= 0:  mos_grade = "Low (0–10%)"
    else:               mos_grade = "Negative — Overvalued"

    max_buy_price = iv * 0.7  # require ≥30% MOS

    # ── Defensive Investor Checklist ─────────────────────────────────────────
    def_checks = [
        {
            "Criterion": "Adequate size (Total Equity > 2,000M LKR)",
            "Condition": f"Equity: {total_equity:,.0f}M",
            "Result": "✅ Pass" if total_equity > 2000 else "❌ Fail",
            "Pass": total_equity > 2000,
        },
        {
            "Criterion": "Strong Liquidity (Current Ratio > 2)" + (" [N/A for banks]" if is_bank else ""),
            "Condition": f"Ratio: {current_ratio:.2f}" if current_ratio > 0 else "N/A",
            "Result": "✅ Pass" if (is_bank or current_ratio > 2) else "❌ Fail",
            "Pass": is_bank or current_ratio > 2,
        },
        {
            "Criterion": "Positive Earnings (EPS > 0)",
            "Condition": f"EPS: {eps:.2f}" + (" (annualized: {:.2f})".format(eps_for_pe) if is_quarterly else ""),
            "Result": "✅ Pass" if eps > 0 else "❌ Fail",
            "Pass": eps > 0,
        },
        {
            "Criterion": "Reasonable P/E (< 15)" + (" annualized" if is_quarterly else ""),
            "Condition": _fmt(pe, 1) if pe else "N/A",
            "Result": "✅ Pass" if (pe and pe < 15) else "❌ Fail",
            "Pass": bool(pe and pe < 15),
        },
        {
            "Criterion": "Reasonable P/B (< 1.5)",
            "Condition": _fmt(pb, 2) if pb else "N/A",
            "Result": "✅ Pass" if (pb and pb < 1.5) else "❌ Fail",
            "Pass": bool(pb and pb < 1.5),
        },
    ]
    passes = sum(1 for c in def_checks if c["Pass"])
    defensive_verdict = "✅ Pass" if passes >= 4 else "❌ Fail"

    # ── Final Buy Decision ────────────────────────────────────────────────────
    if grade in ("A", "B") and mos_pct >= 20:
        buy_decision = "YES — Buy Now"
        alloc_pct = 20 if (grade == "A" and mos_pct >= 30) else 15
    elif grade in ("A", "B") and mos_pct >= 10:
        buy_decision = "YES — Consider Buying"
        alloc_pct = 10
    elif grade in ("B", "C") and mos_pct >= 0:
        buy_decision = "HOLD — Wait for Better Price"
        alloc_pct = 5 if grade == "C" else 8
    else:
        buy_decision = "NO — Avoid / Reduce"
        alloc_pct = 0

    # ── Bottom Line ───────────────────────────────────────────────────────────
    ticker_str = data.get('ticker', 'This stock')
    co_str = data.get('company_name', ticker_str)
    q_note = " (Q)" if is_quarterly else ""
    bank_note = " — a financial institution, so D/E and current ratio are excluded from scoring" if is_bank else ""

    if grade in ("A", "B") and mos_pct >= 20:
        bottom_line = (
            f"{co_str} ({ticker_str}){q_note} earns a Graham Grade {grade} with a {mos_pct:.0f}% "
            f"margin of safety — strong fundamentals at an attractive price.{bank_note}"
        )
    elif grade in ("A", "B"):
        bottom_line = (
            f"{co_str} ({ticker_str}){q_note} has solid fundamentals (Grade {grade}) but limited "
            f"margin of safety ({mos_pct:.0f}%) — wait for a price pullback.{bank_note}"
        )
    elif grade == "C":
        bottom_line = (
            f"{co_str} ({ticker_str}){q_note} scores Grade C — hold existing positions but "
            f"don't add until fundamentals or valuation improve.{bank_note}"
        )
    else:
        bottom_line = (
            f"{co_str} ({ticker_str}){q_note} scores Grade D — weak fundamentals, "
            f"avoid or reduce position.{bank_note}"
        )

    int_score = int(score) if score == int(score) else score

    return {
        # ── Inputs echoed back ──
        "eps": eps, "eps_prev": eps_prev,
        "eps_annualized": eps_for_pe if is_quarterly else None,
        "revenue": revenue, "revenue_prev": revenue_prev,
        "net_income": net_income, "net_income_prev": net_income_prev,
        "total_equity": total_equity, "total_assets": total_assets,
        "total_liabilities": liab,
        "current_assets": current_assets, "current_liabilities": current_liabilities,
        "operating_cf": operating_cf, "shares": shares,
        "market_price": market_price, "dps": dps,
        # ── Flags ──
        "is_quarterly": is_quarterly,
        "is_bank": is_bank,
        # ── Growth ──
        "revenue_growth_pct": revenue_growth,
        "profit_growth_pct":  profit_growth,
        "eps_growth_pct":     eps_growth,
        # ── Valuation ──
        "pe": pe, "pb": pb, "pcf": pcf, "bvps": bvps,
        # ── Profitability ──
        "roe": roe, "roa": roa, "current_ratio": current_ratio, "de": de,
        # ── Dividend ──
        "div_yield": div_yield, "payout_ratio": payout_ratio,
        # ── Score ──
        "graham_score": int_score,
        "score_table": score_table,
        "score_checklist": checklist,
        "graham_grade": grade,
        "recommendation": rec,
        # ── MOS ──
        "intrinsic_value": iv,
        "mos_pct": mos_pct,
        "mos_grade": mos_grade,
        "max_buy_price": max_buy_price,
        # ── Defensive ──
        "defensive_checklist": def_checks,
        "defensive_verdict": defensive_verdict,
        "defensive_passes": passes,
        # ── Final rec ──
        "buy_decision": buy_decision,
        "allocation_pct": alloc_pct,
        "bottom_line": bottom_line,
    }


def calculate_graham_score(data: dict):
    """Backward-compatible 5-tuple: (score, rec, checklist, mos_pct, intrinsic_value)."""
    a = calculate_full_analysis(data)
    return a["graham_score"], a["recommendation"], a["score_checklist"], a["mos_pct"], a["intrinsic_value"]
