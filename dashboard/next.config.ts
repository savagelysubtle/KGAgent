import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Output standalone build for Docker deployment
  // This creates a minimal production build with all dependencies bundled
  output: "standalone",

  // Experimental features
  experimental: {
    // Optimize package imports for faster builds
    optimizePackageImports: ["lucide-react", "@radix-ui/react-icons"],
  },

  // Environment variables that should be available at runtime
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  },
};

export default nextConfig;
