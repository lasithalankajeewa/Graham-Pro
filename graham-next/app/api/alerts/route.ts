import { NextResponse } from "next/server";
import { currentUser } from "@/lib/auth";
import { execute } from "@/lib/db";

export async function POST(req: Request) {
  const username = await currentUser(); if (!username) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  const a = await req.json();
  await execute("INSERT INTO alerts (username,ticker,company_name,alert_type,threshold,email,active) VALUES (?,?,?,?,?,?,1)", [username, String(a.ticker).toUpperCase(), a.company_name, a.alert_type, Number(a.threshold), a.email]);
  return NextResponse.json({ ok: true });
}
export async function DELETE(req: Request) {
  const username = await currentUser(); if (!username) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  const { id } = await req.json(); await execute("DELETE FROM alerts WHERE id=? AND username=?", [Number(id), username]);
  return NextResponse.json({ ok: true });
}
