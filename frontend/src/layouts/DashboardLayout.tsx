import React, { useState } from 'react';
import { Link, useLocation, Outlet } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Activity, 
  Users, 
  BarChart3, 
  Menu, 
  X, 
  Bell, 
  Settings,
  ChevronDown,
  User
} from 'lucide-react';
import { Menu as HeadlessMenu, Transition } from '@headlessui/react';
import clsx from 'clsx';
import StatusDot from '../components/StatusDot';
import { useStats } from '../hooks/usePatients';

interface NavItem {
  name: string;
  path: string;
  icon: typeof Activity;
  role: string;
}

const navigation: NavItem[] = [
  { name: 'Nurse Dashboard', path: '/nurse', icon: Activity, role: 'Nurse' },
  { name: 'Doctor Dashboard', path: '/doctor', icon: Users, role: 'Doctor' },
  { name: 'Chief Dashboard', path: '/chief', icon: BarChart3, role: 'CMO' }
];

export default function DashboardLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const location = useLocation();
  
  // Get real stats from backend
  const { stats } = useStats(10000);
  
  // Determine current role based on path
  const currentPath = location.pathname;
  const currentNavItem = navigation.find(item => item.path === currentPath);
  const currentRole = currentNavItem?.role || 'User';
  
  const toggleSidebar = () => {
    if (window.innerWidth < 768) {
      setSidebarOpen(!sidebarOpen);
    } else {
      setSidebarCollapsed(!sidebarCollapsed);
    }
  };

  return (
    <div className="h-screen flex overflow-hidden bg-gray-100">
      {/* Mobile sidebar backdrop */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-gray-600 bg-opacity-75 z-20 md:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <AnimatePresence>
        <motion.div
          initial={false}
          animate={{ 
            width: (() => {
              if (sidebarOpen || !sidebarCollapsed) {
                return window.innerWidth < 768 ? '280px' : '240px';
              }
              return '64px';
            })(),
            x: sidebarOpen || window.innerWidth >= 768 ? 0 : '-100%'
          }}
          className={clsx(
            'fixed md:relative inset-y-0 left-0 z-30',
            'bg-white border-r border-gray-200',
            'flex flex-col transition-all duration-300'
          )}
        >
          {/* Logo and Toggle */}
          <div className="h-16 flex items-center justify-between px-4 border-b border-gray-200">
            {(!sidebarCollapsed || sidebarOpen) && (
              <Link to="/nurse" className="flex items-center gap-2">
                <Activity className="w-8 h-8 text-primary" />
                <span className="font-bold text-xl text-gray-900">VitalX</span>
              </Link>
            )}
            <button
              onClick={toggleSidebar}
              className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
            >
              {sidebarOpen || !sidebarCollapsed ? (
                <X className="w-5 h-5 text-gray-600" />
              ) : (
                <Menu className="w-5 h-5 text-gray-600" />
              )}
            </button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
            {navigation.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  onClick={() => window.innerWidth < 768 && setSidebarOpen(false)}
                  className={clsx(
                    'flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all',
                    isActive 
                      ? 'bg-primary text-white shadow-md' 
                      : 'text-gray-700 hover:bg-gray-100'
                  )}
                >
                  <Icon className="w-5 h-5 flex-shrink-0" />
                  {(!sidebarCollapsed || sidebarOpen) && (
                    <span className="font-medium">{item.name}</span>
                  )}
                </Link>
              );
            })}
          </nav>

          {/* System Status */}
          {(!sidebarCollapsed || sidebarOpen) && (
            <div className="p-4 border-t border-gray-200">
              <div className="text-xs text-gray-500 mb-2">System Status</div>
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600">Active Patients</span>
                  <span className="font-semibold">{stats?.total_patients || 0}</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600">Critical Alerts</span>
                  <span className="font-semibold text-red-600">{stats?.high_risk_count || 0}</span>
                </div>
                <div className="flex items-center gap-2 text-sm pt-2">
                  <StatusDot status="connected" />
                  <span className="text-gray-600">{stats?.data_source === 'kafka' ? 'Live Data' : 'Mock Data'}</span>
                </div>
              </div>
            </div>
          )}

          {/* Version */}
          <div className="p-4 border-t border-gray-200">
            {(!sidebarCollapsed || sidebarOpen) ? (
              <div className="text-xs text-gray-500">
                Version 1.0.0
              </div>
            ) : (
              <div className="text-xs text-gray-500 text-center">v1</div>
            )}
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top bar */}
        <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-6">
          <div className="flex items-center gap-4">
            <button
              onClick={toggleSidebar}
              className="md:hidden p-2 rounded-lg hover:bg-gray-100"
            >
              <Menu className="w-6 h-6 text-gray-600" />
            </button>
            
            <div>
              <h1 className="text-xl font-bold text-gray-900">{currentNavItem?.name || 'Dashboard'}</h1>
              <p className="text-sm text-gray-500">Real-time VitalX Monitoring</p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {/* Notifications */}
            <button className="relative p-2 rounded-lg hover:bg-gray-100 transition-colors">
              <Bell className="w-5 h-5 text-gray-600" />
              {(stats?.high_risk_count || 0) > 0 && (
                <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full animate-pulse" />
              )}
            </button>

            {/* Settings */}
            <button className="p-2 rounded-lg hover:bg-gray-100 transition-colors">
              <Settings className="w-5 h-5 text-gray-600" />
            </button>

            {/* User menu */}
            <HeadlessMenu as="div" className="relative">
              <HeadlessMenu.Button className="flex items-center gap-2 p-2 rounded-lg hover:bg-gray-100 transition-colors">
                <div className="w-8 h-8 rounded-full bg-primary text-white flex items-center justify-center">
                  <User className="w-5 h-5" />
                </div>
                <div className="hidden md:block text-left">
                  <div className="text-sm font-medium text-gray-900">Current User</div>
                  <div className="text-xs text-gray-500">{currentRole}</div>
                </div>
                <ChevronDown className="w-4 h-4 text-gray-600" />
              </HeadlessMenu.Button>

              <Transition
                enter="transition duration-100 ease-out"
                enterFrom="transform scale-95 opacity-0"
                enterTo="transform scale-100 opacity-100"
                leave="transition duration-75 ease-out"
                leaveFrom="transform scale-100 opacity-100"
                leaveTo="transform scale-95 opacity-0"
              >
                <HeadlessMenu.Items className="absolute right-0 mt-2 w-56 origin-top-right bg-white border border-gray-200 rounded-lg shadow-lg focus:outline-none z-50">
                  <div className="p-3 border-b border-gray-200">
                    <div className="text-sm font-medium text-gray-900">Switch Role</div>
                    <div className="text-xs text-gray-500">Current: {currentRole}</div>
                  </div>
                  
                  <div className="p-2">
                    {navigation.map((item) => (
                      <HeadlessMenu.Item key={item.path}>
                        {({ active }) => (
                          <Link
                            to={item.path}
                            className={clsx(
                              'flex items-center gap-3 px-3 py-2 rounded-md text-sm',
                              active && 'bg-gray-100',
                              location.pathname === item.path && 'text-primary font-medium'
                            )}
                          >
                            {React.createElement(item.icon, { className: 'w-4 h-4' })}
                            {item.role}
                          </Link>
                        )}
                      </HeadlessMenu.Item>
                    ))}
                  </div>
                </HeadlessMenu.Items>
              </Transition>
            </HeadlessMenu>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
