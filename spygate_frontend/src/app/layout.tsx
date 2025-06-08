import { Inter } from "next/font/google";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AuthProvider } from "@/lib/auth";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes
      refetchOnWindowFocus: false,
    },
  },
});

export const metadata = {
  title: "SpygateAI - Game Analysis Platform",
  description:
    "Advanced game analysis platform for EA Sports football games with AI-powered insights.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <QueryClientProvider client={queryClient}>
          <AuthProvider>{children}</AuthProvider>
        </QueryClientProvider>
      </body>
    </html>
  );
}
