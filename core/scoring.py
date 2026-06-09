def calculate_graham_score(data):
    score = 0
    checklist = []

    pe = data.get('pe_ratio', 99)
    if pe < 15:
        score += 3; checklist.append("✅ P/E Ratio < 15 (+3)")
    else:
        checklist.append("❌ P/E Ratio >= 15")

    pb = data.get('pb_ratio', 99)
    if pb < 1.0:
        score += 3; checklist.append("✅ P/B Ratio < 1.0 (+3)")
    elif pb < 1.5:
        score += 2; checklist.append("✅ P/B Ratio < 1.5 (+2)")
    else:
        checklist.append("❌ P/B Ratio >= 1.5")

    growth = data.get('earnings_growth_5yr', 0)
    if growth > 20:
        score += 2; checklist.append("✅ High Growth > 20% (+2)")
    elif growth > 0:
        score += 1; checklist.append("✅ Positive Growth (+1)")
    else:
        checklist.append("❌ Negative/Zero Growth")

    roe = data.get('roe', 0)
    if roe > 15:
        score += 2; checklist.append("✅ High ROE > 15% (+2)")
    elif roe > 10:
        score += 1; checklist.append("✅ Decent ROE > 10% (+1)")
    else:
        checklist.append("❌ Low ROE")

    de = data.get('debt_to_equity', 99)
    if de < 0.5:
        score += 2; checklist.append("✅ Conservative Debt < 0.5 (+2)")
    elif de < 1.0:
        score += 1; checklist.append("✅ Manageable Debt < 1.0 (+1)")
    else:
        checklist.append("❌ High Debt/Equity")

    liabilities = data.get('current_liabilities', 0)
    current_ratio = data.get('current_assets', 0) / liabilities if liabilities > 0 else 0
    if current_ratio > 2.0:
        score += 1; checklist.append("✅ Strong Current Ratio > 2.0 (+1)")

    if data.get('dividend_paid') == 'Yes':
        score += 1; checklist.append("✅ Dividend Payer (+1)")

    if data.get('revenue', 0) > 2000:
        score += 1; checklist.append("✅ Large-Cap Size (+1)")

    if score >= 12: rec = "Strong Buy"
    elif score >= 9: rec = "Buy"
    elif score >= 6: rec = "Hold"
    else: rec = "Sell"

    g = data.get('earnings_growth_5yr', 0)
    v = data.get('eps', 0) * (8.5 + 2 * min(g, 15))
    price = pe * data.get('eps', 1)
    mos = ((v - price) / v * 100) if v > 0 else 0

    return score, rec, checklist, mos, v
