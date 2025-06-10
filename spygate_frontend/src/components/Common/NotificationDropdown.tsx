import React from 'react';
import { Notification } from '../../types';

interface NotificationDropdownProps {
  notifications: Notification[];
  unreadCount: number;
}

const NotificationDropdown: React.FC<NotificationDropdownProps> = ({ 
  notifications, 
  unreadCount 
}) => {
  return (
    <div className="relative">
      <button className="p-2 text-dark-text-muted hover:text-dark-text">
        🔔
        {unreadCount > 0 && (
          <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
            {unreadCount}
          </span>
        )}
      </button>
    </div>
  );
};

export default NotificationDropdown; 