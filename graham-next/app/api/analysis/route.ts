import { NextResponse } from "next/server";
import { currentUser } from "@/lib/auth";
import { insert } from "@/lib/db";
import { calculateAnalysis } from "@/lib/scoring";

export async function POST(req: Request) {
  const username = await currentUser();
  if (!username) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  const input = await req.json();
  if (!String(input.ticker || "").trim() || !String(input.company_name || "").trim()) return NextResponse.json({ error: "Company and ticker are required." }, { status: 400 });
  input.ticker = input.ticker.trim().toUpperCase();
  const analysis = calculateAnalysis(input);
  const data = { ...input, _analysis: analysis };
  const date = new Date().toISOString().slice(0, 16).replace("T", " ");
  const id = await insert("INSERT INTO analysis (username,company_name,ticker,date,data_json,score,recommendation,source) VALUES (?,?,?,?,?,?,?,?)", [username, input.company_name, input.ticker, date, JSON.stringify(data), analysis.graham_score, analysis.recommendation, "manual"]);
  return NextResponse.json({ id, analysis });
}
