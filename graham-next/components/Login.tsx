"use client";
import { FormEvent, useState } from "react";

export default function Login() {
  const [mode, setMode] = useState<"login" | "signup">("login"); const [error, setError] = useState(""); const [busy, setBusy] = useState(false);
  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault(); setBusy(true); setError(""); const fd = new FormData(e.currentTarget);
    const res = await fetch("/api/auth", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ mode, username: fd.get("username"), password: fd.get("password") }) });
    const body = await res.json(); setBusy(false); if (!res.ok) setError(body.error); else location.reload();
  }
  return <main className="auth-page"><section className="auth-copy"><div className="brand"><span>G</span> Graham Pro</div><p className="eyebrow">CSE VALUE INTELLIGENCE</p><h1>Invest with discipline.<br/><em>Decide with clarity.</em></h1><p className="lede">Turn dense financial reports into a focused Graham score, valuation, and margin-of-safety decision.</p><div className="principles"><div><b>15</b><span>point quality score</span></div><div><b>30%</b><span>target safety margin</span></div><div><b>6</b><span>decision workspaces</span></div></div></section><section className="auth-panel"><form onSubmit={submit} className="auth-card"><p className="eyebrow">PRIVATE WORKSPACE</p><h2>{mode === "login" ? "Welcome back" : "Create your account"}</h2><p>Continue to your investment research dashboard.</p><label>Username<input required name="username" autoComplete="username"/></label><label>Password<input required minLength={6} name="password" type="password" autoComplete={mode === "login" ? "current-password" : "new-password"}/></label>{error && <div className="error">{error}</div>}<button className="primary" disabled={busy}>{busy ? "Please wait…" : mode === "login" ? "Enter dashboard" : "Create account"}</button><button type="button" className="text-button" onClick={() => setMode(mode === "login" ? "signup" : "login")}>{mode === "login" ? "New here? Create an account" : "Already have an account? Sign in"}</button></form></section></main>;
}
