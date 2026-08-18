export type FinancialInput = {
  company_name: string; ticker: string; fiscal_year: string; report_type?: string;
  revenue: number; revenue_prev: number; net_income: number; net_income_prev: number;
  eps: number; eps_prev: number; total_assets: number; total_equity: number;
  total_liabilities: number; current_assets: number; current_liabilities: number;
  market_price: number; shares_outstanding: number; dividend_per_share: number;
  operating_cash_flow: number; pe_ratio?: number; pb_ratio?: number; roe?: number;
  debt_to_equity?: number; earnings_growth_5yr?: number;
};

export type ScoreRow = { criterion: string; value: string; points: number; max: number };
export type AnalysisResult = FinancialInput & {
  revenue_growth_pct: number | null; profit_growth_pct: number | null; eps_growth_pct: number | null;
  pe: number | null; pb: number | null; pcf: number | null; roe_calc: number | null;
  roa: number | null; current_ratio: number; de: number | null; div_yield: number;
  payout_ratio: number; graham_score: number; graham_grade: string; recommendation: string;
  intrinsic_value: number; mos_pct: number; mos_grade: string; max_buy_price: number;
  buy_decision: string; allocation_pct: number; bottom_line: string; score_table: ScoreRow[];
};
