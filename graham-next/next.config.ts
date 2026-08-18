import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  serverExternalPackages: ["pg", "pdf-parse"],
  experimental: { serverActions: { bodySizeLimit: "15mb" } },
};

export default nextConfig;
