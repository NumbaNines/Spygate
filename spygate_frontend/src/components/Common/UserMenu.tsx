import React from 'react';
import { User } from '../../types';

interface UserMenuProps {
  user: User | null;
  onLogout: () => void;
}

const UserMenu: React.FC<UserMenuProps> = ({ user, onLogout }) => {
  if (!user) return null;

  return (
    <div className="flex items-center space-x-2">
      <div className="text-sm text-dark-text">
        {user.first_name || user.username}
      </div>
      <button 
        onClick={onLogout}
        className="text-xs text-dark-text-muted hover:text-dark-text"
      >
        Logout
      </button>
    </div>
  );
};

export default UserMenu; 