import { Pool } from "pg";
import path from "node:path";

type Params = Array<string | number | null>;
const url = process.env.DATABASE_URL || "sqlite:///../graham_bot.db";
const postgres = url.startsWith("postgres");
let pool: Pool | undefined;
let sqlite: any;

async function database() {
  if (postgres) {
    pool ||= new Pool({ connectionString: url, ssl: url.includes("localhost") ? false : { rejectUnauthorized: false } });
    return pool;
  }
  if (!sqlite) {
    // Node 22+ built-in SQLite lets local development use the original database file.
    const { DatabaseSync } = await import("node:sqlite");
    const relative = url.replace("sqlite:///", "");
    sqlite = new DatabaseSync(path.resolve(/* turbopackIgnore: true */ process.cwd(), relative));
  }
  return sqlite;
}

function pgSql(sql: string) {
  let i = 0;
  return sql.replace(/\?/g, () => `$${++i}`);
}

export async function query<T = Record<string, unknown>>(sql: string, params: Params = []): Promise<T[]> {
  const db = await database();
  if (postgres) return (await db.query(pgSql(sql), params)).rows as T[];
  return db.prepare(sql).all(...params) as T[];
}

export async function execute(sql: string, params: Params = []) {
  const db = await database();
  if (postgres) return db.query(pgSql(sql), params);
  return db.prepare(sql).run(...params);
}

export async function insert(sql: string, params: Params = []) {
  if (postgres) {
    const result = await query<{ id: number }>(`${sql} RETURNING id`, params);
    return result[0]?.id;
  }
  const result = await execute(sql, params);
  return Number(result.lastInsertRowid);
}

export async function ensureSchema() {
  const identity = postgres ? "INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY" : "INTEGER PRIMARY KEY AUTOINCREMENT";
  await execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)");
  await execute(`CREATE TABLE IF NOT EXISTS analysis (id ${identity}, username TEXT, company_name TEXT, ticker TEXT, date TEXT, data_json TEXT, score REAL, recommendation TEXT, source TEXT DEFAULT 'manual')`);
  await execute(`CREATE TABLE IF NOT EXISTS watchlist (id ${identity}, username TEXT, ticker TEXT, company_name TEXT, added_date TEXT, UNIQUE(username, ticker))`);
  await execute(`CREATE TABLE IF NOT EXISTS alerts (id ${identity}, username TEXT, ticker TEXT, company_name TEXT, alert_type TEXT, threshold REAL, email TEXT, active INTEGER DEFAULT 1, last_triggered TEXT)`);
}
