import { AnalysisResult, FinancialInput, ScoreRow } from "./types";

const n = (v: unknown) => Number(v) || 0;
const pct = (a: number, b: number) => b ? ((a - b) / Math.abs(b)) * 100 : null;
const fmt = (v: number | null, digits = 1, suffix = "") => v == null ? "N/A" : `${v.toFixed(digits)}${suffix}`;

export function calculateAnalysis(input: FinancialInput): AnalysisResult {
  const d = Object.fromEntries(Object.entries(input).map(([k, v]) =>
    typeof v === "number" ? [k, n(v)] : [k, v]
  )) as FinancialInput;
  let eps = n(d.eps), epsPrev = n(d.eps_prev);
  if (!eps && d.net_income > 0 && d.shares_outstanding > 0) eps = d.net_income / d.shares_outstanding;
  if (!epsPrev && d.net_income_prev > 0 && d.shares_outstanding > 0) epsPrev = d.net_income_prev / d.shares_outstanding;
  const quarterly = d.report_type?.toLowerCase() === "quarterly";
  const liabilities = d.total_liabilities || (d.total_assets > d.total_equity ? d.total_assets - d.total_equity : 0);
  const bank = d.total_assets > 0 && liabilities / d.total_assets > .85;
  const annualEps = quarterly ? eps * 4 : eps;
  const bvps = d.total_equity > 0 && d.shares_outstanding > 0 ? d.total_equity / d.shares_outstanding : 0;
  const cfps = d.operating_cash_flow > 0 && d.shares_outstanding > 0 ? d.operating_cash_flow / d.shares_outstanding : 0;
  const pe = d.market_price > 0 && annualEps > 0 ? d.market_price / annualEps : n(d.pe_ratio) || null;
  const pb = d.market_price > 0 && bvps > 0 ? d.market_price / bvps : n(d.pb_ratio) || null;
  const pcf = d.market_price > 0 && cfps > 0 ? d.market_price / cfps : null;
  const roeRaw = d.total_equity > 0 && d.net_income !== 0 ? d.net_income / d.total_equity * 100 : n(d.roe) || null;
  const roe = quarterly && roeRaw != null ? roeRaw * 4 : roeRaw;
  const roaRaw = d.total_assets > 0 && d.net_income !== 0 ? d.net_income / d.total_assets * 100 : null;
  const roa = quarterly && roaRaw != null ? roaRaw * 4 : roaRaw;
  const de = d.total_equity > 0 && liabilities > 0 ? liabilities / d.total_equity : n(d.debt_to_equity) || null;
  const currentRatio = d.current_liabilities > 0 ? d.current_assets / d.current_liabilities : 0;
  const divYield = d.market_price > 0 && d.dividend_per_share > 0 ? d.dividend_per_share / d.market_price * 100 : 0;
  const payout = eps > 0 && d.dividend_per_share > 0 ? d.dividend_per_share / eps * 100 : 0;
  const epsGrowth = pct(eps, epsPrev) ?? (n(d.earnings_growth_5yr) || null);
  const rows: ScoreRow[] = [];
  const add = (criterion: string, value: string, points: number, max: number) => rows.push({ criterion, value, points, max });
  add("P/E below 15", fmt(pe), pe && pe > 0 ? (pe < 15 ? 3 : pe <= 20 ? 1 : 0) : 0, 3);
  add("P/B below 1.0", fmt(pb, 2), pb && pb > 0 ? (pb < 1 ? 3 : pb <= 1.5 ? 2 : 0) : 0, 3);
  add(quarterly ? "YoY EPS growth" : "EPS growth", fmt(epsGrowth, 1, "%"), epsGrowth == null ? 0 : epsGrowth > 20 ? 2 : epsGrowth > 10 ? 1 : quarterly && epsGrowth > 0 ? .5 : 0, 2);
  add(quarterly ? "Annualized ROE" : "ROE", fmt(roe, 1, "%"), roe == null ? 0 : roe > 15 ? 2 : roe > 10 ? 1 : 0, 2);
  if (!bank) {
    add("Debt / equity", fmt(de, 2), de == null ? 0 : quarterly ? (de < .5 ? 2 : 0) : de < .5 ? 2 : de < 1 ? 1 : 0, 2);
    add("Current ratio", fmt(currentRatio, 2), currentRatio > 2 ? 1 : 0, 1);
  }
  add("Dividend yield", fmt(divYield, 1, "%"), divYield > 5 ? 2 : 0, 2);
  const rawScore = rows.reduce((sum, row) => sum + row.points, 0);
  const score = Number.isInteger(rawScore) ? rawScore : Number(rawScore.toFixed(1));
  const grade = bank
    ? score >= 8 ? "A" : score >= 6 ? "B" : score >= 4 ? "C" : "D"
    : score >= 10 ? "A" : score >= 7 ? "B" : score >= 4 ? "C" : "D";
  const recommendation = ({ A: "Strong Buy", B: "Buy", C: "Hold", D: "Sell" } as Record<string, string>)[grade];
  const intrinsic = annualEps > 0 ? annualEps * 7.4 : 0;
  const mos = intrinsic > 0 && d.market_price > 0 ? (intrinsic - d.market_price) / intrinsic * 100 : 0;
  const mosGrade = mos >= 30 ? "Excellent (>30%)" : mos >= 20 ? "Good (20–30%)" : mos >= 10 ? "Moderate (10–20%)" : mos >= 0 ? "Low (0–10%)" : "Negative — Overvalued";
  const buy = ["A", "B"].includes(grade) && mos >= 20 ? "YES — Buy Now" : ["A", "B"].includes(grade) && mos >= 10 ? "YES — Consider Buying" : ["B", "C"].includes(grade) && mos >= 0 ? "HOLD — Wait for Better Price" : "NO — Avoid / Reduce";
  const allocation = buy.includes("Buy Now") ? (grade === "A" && mos >= 30 ? 20 : 15) : buy.includes("Consider") ? 10 : buy.includes("HOLD") ? (grade === "C" ? 5 : 8) : 0;
  const bottom = `${d.company_name} (${d.ticker}) earns Graham Grade ${grade} with a ${mos.toFixed(0)}% margin of safety. ${buy}.`;
  return { ...d, eps, eps_prev: epsPrev, total_liabilities: liabilities,
    revenue_growth_pct: pct(d.revenue, d.revenue_prev), profit_growth_pct: pct(d.net_income, d.net_income_prev), eps_growth_pct: epsGrowth,
    pe, pb, pcf, roe_calc: roe, roa, current_ratio: currentRatio, de, div_yield: divYield, payout_ratio: payout,
    graham_score: score, graham_grade: grade, recommendation, intrinsic_value: intrinsic, mos_pct: mos,
    mos_grade: mosGrade, max_buy_price: intrinsic * .7, buy_decision: buy, allocation_pct: allocation,
    bottom_line: bottom, score_table: rows };
}
