/**
 * SpygateAI Frontend Type Definitions
 * Matches Django backend models and API responses
 */

// ============================================================================
// User & Authentication Types
// ============================================================================

export interface User {
  id: number;
  username: string;
  email: string;
  first_name: string;
  last_name: string;
  avatar?: string;
  is_premium: boolean;
  tier: UserTier;
  created_at: string;
  last_login: string;
}

export enum UserTier {
  FREE = 'free',
  PREMIUM = 'premium',
  PROFESSIONAL = 'professional',
}

export interface AuthTokens {
  access: string;
  refresh: string;
}

export interface LoginCredentials {
  username: string;
  password: string;
}

export interface RegisterData {
  username: string;
  email: string;
  password: string;
  first_name?: string;
  last_name?: string;
}

// ============================================================================
// Game & Multi-Game Architecture Types
// ============================================================================

export enum GameVersion {
  MADDEN_25 = 'madden_25',
  CFB_25 = 'cfb_25',
  MADDEN_26 = 'madden_26',
  UNIVERSAL = 'universal',
}

export interface GameProfile {
  id: number;
  name: string;
  version: GameVersion;
  display_name: string;
  is_active: boolean;
  supported_features: string[];
  hud_config: Record<string, any>;
  detection_models: string[];
}

// ============================================================================
// Analysis & Strategy Types
// ============================================================================

export interface VideoAnalysis {
  id: number;
  user: number;
  title: string;
  game_version: GameVersion;
  video_file?: string;
  duration: number;
  status: AnalysisStatus;
  results: AnalysisResult[];
  created_at: string;
  updated_at: string;
  tags: string[];
}

export enum AnalysisStatus {
  PENDING = 'pending',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed',
}

export interface AnalysisResult {
  id: number;
  timestamp: number;
  situation: GameSituation;
  formation?: FormationData;
  play_result?: PlayResult;
  confidence: number;
  notes?: string;
}

export interface GameSituation {
  down: number;
  distance: number;
  yard_line: number;
  quarter: number;
  time_remaining: string;
  score_differential: number;
  field_position: 'red_zone' | 'midfield' | 'own_territory';
}

// ============================================================================
// Formation Analysis Types
// ============================================================================

export enum FormationType {
  // Offensive formations
  SPREAD = 'spread',
  I_FORMATION = 'i_formation',
  SHOTGUN = 'shotgun',
  PISTOL = 'pistol',
  UNDER_CENTER = 'under_center',
  WILDCAT = 'wildcat',
  TRIPS = 'trips',
  BUNCH = 'bunch',
  EMPTY = 'empty',
  GOAL_LINE = 'goal_line',

  // Defensive formations
  FOUR_THREE = '4-3',
  THREE_FOUR = '3-4',
  NICKEL = 'nickel',
  DIME = 'dime',
  QUARTER = 'quarter',
  SIX_ONE = '6-1',
  FIVE_TWO = '5-2',
  FOUR_FOUR = '4-4',
  BEAR = 'bear',
  COVER_TWO = 'cover_2',
  COVER_THREE = 'cover_3',
  COVER_FOUR = 'cover_4',

  // Special formations
  SPECIAL_TEAMS = 'special_teams',
  PUNT = 'punt',
  KICK_RETURN = 'kick_return',
  FIELD_GOAL = 'field_goal',

  UNKNOWN = 'unknown',
}

export interface FormationData {
  formation_type: FormationType;
  confidence: number;
  player_positions: PlayerPosition[];
  clusters: ClusterData[];
  is_offense: boolean;
}

export interface PlayerPosition {
  x: number;
  y: number;
  confidence: number;
  player_type?: string;
}

export interface ClusterData {
  center_x: number;
  center_y: number;
  player_count: number;
  cluster_type: string;
}

// ============================================================================
// Performance & Benchmarking Types
// ============================================================================

export enum PerformanceTier {
  CLUTCH = 'clutch',           // 95-100 points
  BIG_PLAY = 'big_play',       // 85-94 points
  GOOD_PLAY = 'good_play',     // 75-84 points
  AVERAGE = 'average',         // 60-74 points
  POOR_PLAY = 'poor_play',     // 40-59 points
  TURNOVER = 'turnover',       // 0-39 points
  DEFENSIVE_STAND = 'defensive_stand', // 0-20 points
}

export interface PlayResult {
  yards_gained: number;
  first_down: boolean;
  touchdown: boolean;
  turnover: boolean;
  safety: boolean;
  performance_tier: PerformanceTier;
  epa: number; // Expected Points Added
  win_probability_change: number;
  leverage_index: number;
}

export interface PerformanceStats {
  total_plays: number;
  avg_performance_score: number;
  tier_distribution: Record<PerformanceTier, number>;
  improvement_trend: number;
  strength_areas: string[];
  weakness_areas: string[];
}

// ============================================================================
// Strategy & Gameplan Types
// ============================================================================

export interface Strategy {
  id: number;
  user: number;
  title: string;
  description: string;
  game_versions: GameVersion[];
  formation_type?: FormationType;
  situation_tags: string[];
  success_rate?: number;
  usage_count: number;
  is_public: boolean;
  parent_strategy?: number; // For cross-game mappings
  created_at: string;
  updated_at: string;
}

export interface Gameplan {
  id: number;
  user: number;
  title: string;
  opponent_username?: string;
  game_version: GameVersion;
  strategies: Strategy[];
  notes: string;
  is_tournament_prep: boolean;
  created_at: string;
  updated_at: string;
}

export interface OpponentAnalysis {
  username: string;
  game_versions: GameVersion[];
  total_games_analyzed: number;
  tendencies: FormationTendency[];
  weaknesses: string[];
  counters: Strategy[];
  success_rate_against: number;
}

export interface FormationTendency {
  formation_type: FormationType;
  usage_percentage: number;
  situations: GameSituation[];
  success_rate: number;
  counter_strategies: Strategy[];
}

// ============================================================================
// Community & Collaboration Types
// ============================================================================

export interface Team {
  id: number;
  name: string;
  description: string;
  owner: User;
  members: User[];
  game_versions: GameVersion[];
  is_private: boolean;
  created_at: string;
}

export interface SharedAnalysis {
  id: number;
  analysis: VideoAnalysis;
  shared_by: User;
  team?: Team;
  is_public: boolean;
  view_count: number;
  likes: number;
  comments: Comment[];
  shared_at: string;
}

export interface Comment {
  id: number;
  user: User;
  content: string;
  parent?: number;
  created_at: string;
  updated_at: string;
}

// ============================================================================
// Real-time & Notification Types
// ============================================================================

export interface Notification {
  id: number;
  user: number;
  type: NotificationType;
  title: string;
  message: string;
  data?: Record<string, any>;
  is_read: boolean;
  created_at: string;
}

export enum NotificationType {
  ANALYSIS_COMPLETE = 'analysis_complete',
  STRATEGY_SHARED = 'strategy_shared',
  TEAM_INVITE = 'team_invite',
  COMMENT_REPLY = 'comment_reply',
  SYSTEM_UPDATE = 'system_update',
  TOURNAMENT_REMINDER = 'tournament_reminder',
}

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: string;
}

// ============================================================================
// API Response Types
// ============================================================================

export interface ApiResponse<T = any> {
  data: T;
  message?: string;
  status: number;
}

export interface PaginatedResponse<T> {
  count: number;
  next?: string;
  previous?: string;
  results: T[];
}

export interface ApiError {
  detail: string;
  code?: string;
  field_errors?: Record<string, string[]>;
}

// ============================================================================
// UI & Component Types
// ============================================================================

export interface TabItem {
  id: string;
  label: string;
  icon?: React.ComponentType;
  badge?: number;
}

export interface FilterOption {
  value: string;
  label: string;
  count?: number;
}

export interface SortOption {
  value: string;
  label: string;
  direction: 'asc' | 'desc';
}

export interface ProgressCallback {
  (status: string, progress: number): void;
}

// ============================================================================
// Cross-Game Intelligence Types
// ============================================================================

export interface UniversalConcept {
  id: string;
  name: string;
  description: string;
  game_mappings: Record<GameVersion, string>;
  effectiveness_data: Record<GameVersion, number>;
  transferability_score: number;
}

export interface CrossGameInsight {
  concept: UniversalConcept;
  source_games: GameVersion[];
  target_game: GameVersion;
  confidence: number;
  recommendations: string[];
}
