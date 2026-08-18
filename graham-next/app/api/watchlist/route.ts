import { NextResponse } from "next/server";
import { currentUser } from "@/lib/auth";
import { execute } from "@/lib/db";

export async function POST(req: Request) {
  const username = await currentUser(); if (!username) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  const { ticker, company_name } = await req.json();
  try { await execute("INSERT INTO watchlist (username,ticker,company_name,added_date) VALUES (?,?,?,?)", [username, String(ticker).toUpperCase(), company_name, new Date().toISOString().slice(0, 10)]); } catch {}
  return NextResponse.json({ ok: true });
}
export async function DELETE(req: Request) {
  const username = await currentUser(); if (!username) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  const { ticker } = await req.json(); await execute("DELETE FROM watchlist WHERE username=? AND ticker=?", [username, ticker]);
  return NextResponse.json({ ok: true });
}
