import { NextResponse } from "next/server";
import { currentUser } from "@/lib/auth";
import { ensureSchema, query } from "@/lib/db";

export async function GET() {
  const username = await currentUser();
  if (!username) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  await ensureSchema();
  const [history, watchlist, alerts] = await Promise.all([
    query("SELECT * FROM analysis WHERE username=? OR source='auto' ORDER BY date DESC", [username]),
    query("SELECT * FROM watchlist WHERE username=? ORDER BY added_date DESC", [username]),
    query("SELECT * FROM alerts WHERE username=? ORDER BY id DESC", [username]),
  ]);
  return NextResponse.json({ history, watchlist, alerts });
}
