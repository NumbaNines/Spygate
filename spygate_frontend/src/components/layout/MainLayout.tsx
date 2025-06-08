import React from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import {
  HomeIcon,
  ChartBarIcon,
  VideoCameraIcon,
  UserIcon,
  ArrowLeftOnRectangleIcon,
} from "@heroicons/react/24/outline";

interface MainLayoutProps {
  children: React.ReactNode;
}

export function MainLayout({ children }: MainLayoutProps) {
  const { user, logout } = useAuth();
  const router = useRouter();

  if (!user) {
    router.push("/login");
    return null;
  }

  const navigation = [
    { name: "Dashboard", href: "/dashboard", icon: HomeIcon },
    { name: "Analysis", href: "/analysis", icon: ChartBarIcon },
    { name: "Videos", href: "/videos", icon: VideoCameraIcon },
    { name: "Profile", href: "/profile", icon: UserIcon },
  ];

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Sidebar */}
      <div className="fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg">
        <div className="flex h-full flex-col">
          {/* Logo */}
          <div className="flex h-16 items-center justify-center border-b border-gray-200">
            <Link href="/dashboard" className="text-xl font-bold text-gray-900">
              SpygateAI
            </Link>
          </div>

          {/* Navigation */}
          <nav className="flex-1 space-y-1 px-2 py-4">
            {navigation.map((item) => (
              <Link
                key={item.name}
                href={item.href}
                className="group flex items-center rounded-md px-2 py-2 text-sm font-medium text-gray-600 hover:bg-gray-50 hover:text-gray-900"
              >
                <item.icon
                  className="mr-3 h-6 w-6 flex-shrink-0 text-gray-400 group-hover:text-gray-500"
                  aria-hidden="true"
                />
                {item.name}
              </Link>
            ))}
          </nav>

          {/* User section */}
          <div className="border-t border-gray-200 p-4">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-gray-500">
                  <span className="text-sm font-medium leading-none text-white">
                    {user.first_name?.[0] || user.username[0]}
                  </span>
                </span>
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium text-gray-700">
                  {user.first_name
                    ? `${user.first_name} ${user.last_name}`
                    : user.username}
                </p>
                <button
                  onClick={logout}
                  className="group flex items-center text-sm font-medium text-gray-500 hover:text-gray-700"
                >
                  <ArrowLeftOnRectangleIcon
                    className="mr-2 h-5 w-5 text-gray-400 group-hover:text-gray-500"
                    aria-hidden="true"
                  />
                  Logout
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="pl-64">
        <main className="py-6">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 md:px-8">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
