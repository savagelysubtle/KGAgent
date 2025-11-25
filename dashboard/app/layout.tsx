import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/app-sidebar";
import { Toaster } from "@/components/ui/sonner";
import { CopilotProviderWithHistory } from "@/components/chat-with-history";
import { Providers } from "@/components/providers";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "KG Agent Dashboard",
  description: "Control plane for Web-to-KG ETL Pipeline",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-gradient-to-br from-black via-slate-900 to-purple-950 min-h-screen text-foreground`}
        suppressHydrationWarning
      >
        <Providers>
          <CopilotProviderWithHistory>
            <SidebarProvider>
            <AppSidebar />
            <main className="w-full min-h-screen relative flex flex-col">
              <div className="p-4">
                  <SidebarTrigger />
              </div>
              <div className="flex-1 p-4 md:p-8 pt-0">
        {children}
              </div>
            </main>
              <Toaster />
            </SidebarProvider>
          </CopilotProviderWithHistory>
        </Providers>
      </body>
    </html>
  );
}
