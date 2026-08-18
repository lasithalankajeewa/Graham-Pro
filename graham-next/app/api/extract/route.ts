import { NextResponse } from "next/server";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { currentUser } from "@/lib/auth";

export const runtime = "nodejs";

const prompt = `Extract financial data from this annual or quarterly report text. Return JSON only, with keys:
company_name, ticker, fiscal_year, report_type (annual or quarterly), revenue, revenue_prev,
net_income, net_income_prev, eps, eps_prev, total_assets, total_equity, total_liabilities,
current_assets, current_liabilities, market_price, shares_outstanding, dividend_per_share,
operating_cash_flow, pe_ratio, pb_ratio, roe, debt_to_equity, earnings_growth_5yr.
All financial totals and shares must be in millions. Use 0 for unavailable numeric values.`;

function jsonFrom(text: string) {
  const clean = text.replace(/```json|```/g, "").trim();
  const start = clean.indexOf("{"); const end = clean.lastIndexOf("}");
  if (start < 0 || end < start) throw new Error("The model did not return valid JSON.");
  return JSON.parse(clean.slice(start, end + 1));
}

export async function POST(req: Request) {
  if (!(await currentUser())) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  const form = await req.formData(); const file = form.get("file");
  if (!(file instanceof File) || file.type !== "application/pdf") return NextResponse.json({ error: "Please upload a PDF." }, { status: 400 });
  if (file.size > 14 * 1024 * 1024) return NextResponse.json({ error: "PDF must be smaller than 14 MB." }, { status: 413 });
  try {
    const bytes = Buffer.from(await file.arrayBuffer());
    let answer = "";
    if (process.env.GEMINI_API_KEY) {
      const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
      const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
      answer = (await model.generateContent([prompt, { inlineData: { mimeType: "application/pdf", data: bytes.toString("base64") } }])).response.text();
    } else if (process.env.OPENROUTER_API_KEY) {
      const pdfParse = (await import("pdf-parse")).default;
      const parsed = await pdfParse(bytes);
      const response = await fetch("https://openrouter.ai/api/v1/chat/completions", { method: "POST", headers: { Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`, "Content-Type": "application/json" }, body: JSON.stringify({ model: "google/gemma-4-31b-it:free", response_format: { type: "json_object" }, temperature: 0, messages: [{ role: "user", content: `${prompt}\n\nREPORT:\n${parsed.text.slice(0, 100000)}` }] }) });
      if (!response.ok) throw new Error(`OpenRouter error ${response.status}`);
      answer = (await response.json()).choices?.[0]?.message?.content || "";
    } else return NextResponse.json({ error: "Set GEMINI_API_KEY or OPENROUTER_API_KEY." }, { status: 503 });
    return NextResponse.json({ data: jsonFrom(answer) });
  } catch (error) { return NextResponse.json({ error: error instanceof Error ? error.message : "Extraction failed." }, { status: 500 }); }
}
