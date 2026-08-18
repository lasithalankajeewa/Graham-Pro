import { currentUser } from "@/lib/auth";
import Dashboard from "@/components/Dashboard";
import Login from "@/components/Login";

export default async function Home() {
  const user = await currentUser();
  return user ? <Dashboard username={user} /> : <Login />;
}
