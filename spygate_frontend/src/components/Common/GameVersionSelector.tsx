import React from 'react';
import { GameVersion } from '../../types';

interface GameVersionSelectorProps {
  selectedGame: GameVersion;
  onGameChange: (game: GameVersion) => void;
}

const GameVersionSelector: React.FC<GameVersionSelectorProps> = ({ 
  selectedGame, 
  onGameChange 
}) => {
  return (
    <select 
      value={selectedGame} 
      onChange={(e) => onGameChange(e.target.value as GameVersion)}
      className="bg-dark-elevated text-dark-text border border-dark-border rounded px-3 py-1"
    >
      <option value={GameVersion.MADDEN_25}>Madden 25</option>
      <option value={GameVersion.CFB_25}>CFB 25</option>
      <option value={GameVersion.UNIVERSAL}>Universal</option>
    </select>
  );
};

export default GameVersionSelector; 