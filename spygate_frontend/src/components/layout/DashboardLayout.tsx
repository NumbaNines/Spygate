/**
 * Main Dashboard Layout Component
 * FACEIT-style dark theme with SpygateAI branding
 */

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import {
  HomeIcon,
  VideoCameraIcon,
  ChartBarIcon,
  UserGroupIcon,
  Cog6ToothIcon,
  BellIcon,
  PlayIcon,
  TrophyIcon,
  AcademicCapIcon,
  ArrowRightOnRectangleIcon,
  Bars3Icon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import { BellIcon as BellSolidIcon } from '@heroicons/react/24/solid';
import { User, Notification, GameVersion } from '../../types';
import { useAuth } from '../../hooks/useAuth';
import { useNotifications } from '../../hooks/useNotifications';
import GameVersionSelector from '../Common/GameVersionSelector';
import NotificationDropdown from '../Common/NotificationDropdown';
import UserMenu from '../Common/UserMenu';

interface DashboardLayoutProps {
  children: React.ReactNode;
  title?: string;
  subtitle?: string;
}

interface NavItem {
  name: string;
  href: string;
  icon: React.ComponentType<any>;
  current?: boolean;
  badge?: number;
  description?: string;
}

const DashboardLayout: React.FC<DashboardLayoutProps> = ({
  children,
  title = 'Dashboard',
  subtitle
}) => {
  const router = useRouter();
  const { user, logout } = useAuth();
  const { notifications, unreadCount } = useNotifications();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedGame, setSelectedGame] = useState<GameVersion>(GameVersion.MADDEN_25);

  // Navigation items
  const navigation: NavItem[] = [
    {
      name: 'Dashboard',
      href: '/dashboard',
      icon: HomeIcon,
      current: router.pathname === '/dashboard',
      description: 'Overview and quick stats'
    },
    {
      name: 'Video Analysis',
      href: '/analysis',
      icon: VideoCameraIcon,
      current: router.pathname.startsWith('/analysis'),
      description: 'Upload and analyze gameplay'
    },
    {
      name: 'Strategies',
      href: '/strategies',
      icon: PlayIcon,
      current: router.pathname.startsWith('/strategies'),
      description: 'Manage your gameplans'
    },
    {
      name: 'Performance',
      href: '/performance',
      icon: ChartBarIcon,
      current: router.pathname.startsWith('/performance'),
      description: '7-tier performance tracking'
    },
    {
      name: 'Opponents',
      href: '/opponents',
      icon: TrophyIcon,
      current: router.pathname.startsWith('/opponents'),
      description: 'Scout and analyze opponents'
    },
    {
      name: 'Teams',
      href: '/teams',
      icon: UserGroupIcon,
      current: router.pathname.startsWith('/teams'),
      description: 'Collaborate with your team'
    },
    {
      name: 'Learning Hub',
      href: '/learning',
      icon: AcademicCapIcon,
      current: router.pathname.startsWith('/learning'),
      description: 'Pro strategies and guides'
    },
  ];

  const secondaryNavigation: NavItem[] = [
    {
      name: 'Settings',
      href: '/settings',
      icon: Cog6ToothIcon,
      current: router.pathname.startsWith('/settings'),
    },
  ];

  return (
    <div className="min-h-screen bg-dark-bg">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div className="fixed inset-0 z-50 lg:hidden">
          <div className="fixed inset-0 bg-black/50" onClick={() => setSidebarOpen(false)} />
          <div className="fixed inset-y-0 left-0 w-64 bg-dark-surface border-r border-dark-border">
            <SidebarContent
              navigation={navigation}
              secondaryNavigation={secondaryNavigation}
              user={user}
              onLogout={logout}
              onClose={() => setSidebarOpen(false)}
              isMobile
            />
          </div>
        </div>
      )}

      {/* Desktop sidebar */}
      <div className="hidden lg:fixed lg:inset-y-0 lg:left-0 lg:z-40 lg:w-64">
        <div className="flex h-full flex-col bg-dark-surface border-r border-dark-border">
          <SidebarContent
            navigation={navigation}
            secondaryNavigation={secondaryNavigation}
            user={user}
            onLogout={logout}
          />
        </div>
      </div>

      {/* Main content area */}
      <div className="lg:pl-64">
        {/* Top navigation bar */}
        <div className="sticky top-0 z-30 bg-dark-surface/95 backdrop-blur border-b border-dark-border">
          <div className="flex h-16 items-center justify-between px-4 sm:px-6 lg:px-8">
            {/* Mobile menu button */}
            <button
              type="button"
              className="lg:hidden -m-2.5 p-2.5 text-dark-text"
              onClick={() => setSidebarOpen(true)}
            >
              <Bars3Icon className="h-6 w-6" />
            </button>

            {/* Page title */}
            <div className="flex-1 lg:flex-none">
              <h1 className="text-xl font-semibold text-dark-text">{title}</h1>
              {subtitle && (
                <p className="text-sm text-dark-text-muted">{subtitle}</p>
              )}
            </div>

            {/* Top navigation controls */}
            <div className="flex items-center space-x-4">
              {/* Game version selector */}
              <GameVersionSelector
                selectedGame={selectedGame}
                onGameChange={setSelectedGame}
              />

              {/* Notifications */}
              <NotificationDropdown
                notifications={notifications}
                unreadCount={unreadCount}
              />

              {/* User menu */}
              <UserMenu user={user} onLogout={logout} />
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className="min-h-[calc(100vh-4rem)]">
          {children}
        </main>
      </div>
    </div>
  );
};

// Sidebar content component (reused for mobile and desktop)
interface SidebarContentProps {
  navigation: NavItem[];
  secondaryNavigation: NavItem[];
  user: User | null;
  onLogout: () => void;
  onClose?: () => void;
  isMobile?: boolean;
}

const SidebarContent: React.FC<SidebarContentProps> = ({
  navigation,
  secondaryNavigation,
  user,
  onLogout,
  onClose,
  isMobile = false,
}) => {
  return (
    <>
      {/* SpygateAI Logo */}
      <div className="flex h-16 items-center px-6 border-b border-dark-border">
        {isMobile && (
          <button
            onClick={onClose}
            className="mr-4 p-1 text-dark-text-muted hover:text-dark-text"
          >
            <XMarkIcon className="h-5 w-5" />
          </button>
        )}
        <div className="flex items-center">
          <div className="flex-shrink-0">
            <div className="h-8 w-8 bg-spygate-orange rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">SG</span>
            </div>
          </div>
          <div className="ml-3">
            <h2 className="text-lg font-semibold text-dark-text">SpygateAI</h2>
            <p className="text-xs text-dark-text-muted">Pro Football Analysis</p>
          </div>
        </div>
      </div>

      {/* User tier badge */}
      {user && (
        <div className="px-6 py-3 border-b border-dark-border">
          <div className="flex items-center">
            <div className="flex-shrink-0">
              <div className="h-8 w-8 rounded-full bg-dark-elevated flex items-center justify-center">
                <span className="text-sm font-medium text-dark-text">
                  {user.first_name?.[0] || user.username[0].toUpperCase()}
                </span>
              </div>
            </div>
            <div className="ml-3 min-w-0 flex-1">
              <p className="text-sm font-medium text-dark-text truncate">
                {user.first_name || user.username}
              </p>
              <div className="flex items-center mt-1">
                <span className={`
                  inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium
                  ${user.tier === 'professional'
                    ? 'bg-tier-clutch/20 text-tier-clutch'
                    : user.tier === 'premium'
                    ? 'bg-tier-big/20 text-tier-big'
                    : 'bg-tier-average/20 text-tier-average'
                  }
                `}>
                  {user.tier.charAt(0).toUpperCase() + user.tier.slice(1)}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {navigation.map((item) => (
          <Link
            key={item.name}
            href={item.href}
            onClick={onClose}
            className={`
              group flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors
              ${item.current
                ? 'bg-spygate-orange text-white'
                : 'text-dark-text-secondary hover:text-dark-text hover:bg-dark-elevated'
              }
            `}
          >
            <item.icon
              className={`
                mr-3 h-5 w-5 flex-shrink-0
                ${item.current ? 'text-white' : 'text-dark-text-muted'}
              `}
            />
            <span className="flex-1">{item.name}</span>
            {item.badge && item.badge > 0 && (
              <span className="ml-2 inline-flex items-center justify-center px-2 py-1 text-xs font-bold leading-none text-white bg-red-500 rounded-full">
                {item.badge}
              </span>
            )}
          </Link>
        ))}

        {/* Divider */}
        <div className="border-t border-dark-border my-4" />

        {/* Secondary navigation */}
        {secondaryNavigation.map((item) => (
          <Link
            key={item.name}
            href={item.href}
            onClick={onClose}
            className={`
              group flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors
              ${item.current
                ? 'bg-spygate-orange text-white'
                : 'text-dark-text-secondary hover:text-dark-text hover:bg-dark-elevated'
              }
            `}
          >
            <item.icon
              className={`
                mr-3 h-5 w-5 flex-shrink-0
                ${item.current ? 'text-white' : 'text-dark-text-muted'}
              `}
            />
            {item.name}
          </Link>
        ))}
      </nav>

      {/* Bottom actions */}
      <div className="border-t border-dark-border p-3">
        <button
          onClick={() => {
            onLogout();
            onClose?.();
          }}
          className="w-full flex items-center px-3 py-2 text-sm font-medium text-dark-text-secondary hover:text-dark-text hover:bg-dark-elevated rounded-lg transition-colors"
        >
          <ArrowRightOnRectangleIcon className="mr-3 h-5 w-5 text-dark-text-muted" />
          Sign Out
        </button>
      </div>
    </>
  );
};

export default DashboardLayout;
