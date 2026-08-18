import { NextResponse } from "next/server";
import bcrypt from "bcryptjs";
import { createSession, cookieName } from "@/lib/auth";
import { ensureSchema, execute, query } from "@/lib/db";

export async function POST(req: Request) {
  const { mode, username: raw, password } = await req.json();
  const username = String(raw || "").trim();
  if (username.length < 2 || String(password || "").length < 6) return NextResponse.json({ error: "Use a valid username and a password of at least 6 characters." }, { status: 400 });
  await ensureSchema();
  const users = await query<{ username: string; password: string }>("SELECT username, password FROM users WHERE username=?", [username]);
  if (mode === "signup") {
    if (users.length) return NextResponse.json({ error: "Username already exists." }, { status: 409 });
    await execute("INSERT INTO users (username,password) VALUES (?,?)", [username, await bcrypt.hash(password, 12)]);
  } else if (!users[0] || !(await bcrypt.compare(password, users[0].password))) {
    return NextResponse.json({ error: "Invalid username or password." }, { status: 401 });
  }
  await createSession(username);
  return NextResponse.json({ ok: true });
}

export async function DELETE() {
  const response = NextResponse.json({ ok: true });
  response.cookies.set(cookieName, "", { expires: new Date(0), path: "/" });
  return response;
}
