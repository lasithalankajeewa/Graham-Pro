import { SignJWT, jwtVerify } from "jose";
import { cookies } from "next/headers";

const secret = new TextEncoder().encode(process.env.SECRET_KEY || "development-only-change-me");
export const cookieName = "graham_session";

export async function createSession(username: string) {
  const token = await new SignJWT({ username }).setProtectedHeader({ alg: "HS256" }).setIssuedAt().setExpirationTime("7d").sign(secret);
  (await cookies()).set(cookieName, token, { httpOnly: true, sameSite: "lax", secure: process.env.NODE_ENV === "production", path: "/", maxAge: 604800 });
}

export async function currentUser() {
  try {
    const token = (await cookies()).get(cookieName)?.value;
    if (!token) return null;
    const { payload } = await jwtVerify(token, secret);
    return typeof payload.username === "string" ? payload.username : null;
  } catch { return null; }
}
