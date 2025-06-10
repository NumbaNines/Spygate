# SpygateAI Frontend

A Next.js React frontend for the SpygateAI web collaboration platform.

## Overview

This is the web collaboration hub component of SpygateAI, designed to complement the desktop PyQt6 application. It provides:

- **Community Dashboard**: Strategy sharing and collaboration
- **Team Management**: Create and join teams for tournament prep
- **Cross-Game Intelligence**: Strategy migration between EA football games
- **Real-time Collaboration**: WebSocket-powered real-time features
- **Mobile-Friendly**: Responsive design for on-the-go strategy review

## Architecture

### Design Philosophy
- **FACEIT-style UI**: Dark theme with professional gaming aesthetics
- **Performance First**: Optimized for fast loading and smooth interactions
- **Mobile-Responsive**: Works seamlessly on desktop, tablet, and mobile
- **API-Driven**: Clean separation between frontend and Django backend

### Technology Stack
- **Framework**: Next.js 14 with TypeScript
- **Styling**: Tailwind CSS with custom SpygateAI theme
- **State Management**: React Query for server state, React Context for app state
- **Authentication**: JWT tokens with automatic refresh
- **Real-time**: Socket.io for WebSocket communication
- **Forms**: React Hook Form with Zod validation

## Getting Started

### Prerequisites
- Node.js 18.0.0 or higher
- npm or yarn
- Django backend running on `http://localhost:8000`

### Installation

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env.local
   ```

   Configure the following variables:
   ```env
   NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
   NEXT_PUBLIC_WS_URL=ws://localhost:8000
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

4. **Open browser**:
   Navigate to `http://localhost:3000`

### Development Commands

```bash
# Development server
npm run dev

# Production build
npm run build

# Start production server
npm start

# Type checking
npm run type-check

# Linting
npm run lint
```

## Project Structure

```
spygate_frontend/
├── pages/                  # Next.js pages (routing)
│   ├── _app.tsx           # App wrapper with providers
│   ├── index.tsx          # Homepage (redirects to dashboard)
│   ├── dashboard/         # Dashboard pages
│   ├── analysis/          # Video analysis pages
│   ├── strategies/        # Strategy management pages
│   ├── teams/             # Team collaboration pages
│   └── auth/              # Authentication pages
├── src/
│   ├── components/        # React components
│   │   ├── Layout/        # Layout components
│   │   ├── Common/        # Reusable components
│   │   ├── Dashboard/     # Dashboard-specific components
│   │   ├── Analysis/      # Video analysis components
│   │   ├── Strategy/      # Strategy components
│   │   └── Team/          # Team components
│   ├── hooks/             # Custom React hooks
│   ├── lib/               # Utility libraries
│   │   ├── api.ts         # API client
│   │   ├── utils.ts       # General utilities
│   │   └── websocket.ts   # WebSocket client
│   ├── types/             # TypeScript type definitions
│   └── styles/            # Global styles
├── public/                # Static assets
└── tailwind.config.js     # Tailwind configuration
```

## Key Features

### 1. Community Dashboard
- View shared strategies from the community
- Browse pro player analysis
- Filter by game version and strategy type
- Like and comment on shared content

### 2. Strategy Management
- Create and organize personal strategies
- Cross-game strategy migration
- Share strategies with teams or community
- Import strategies from desktop app

### 3. Team Collaboration
- Create teams for tournament preparation
- Invite team members
- Share private analysis and strategies
- Real-time collaboration features

### 4. Opponent Analysis
- Community-powered opponent scouting
- Historical performance data
- Shared opponent strategies
- Tournament preparation tools

### 5. Performance Tracking
- 7-tier performance scoring system
- Cross-game performance comparison
- Improvement trend analysis
- Professional benchmarking

## Integration with Desktop App

The frontend integrates seamlessly with the PyQt6 desktop application:

- **API Bridge**: Desktop app can push/pull data to/from web platform
- **Strategy Sync**: Automatic synchronization of strategies and analysis
- **Cloud Backup**: Optional cloud storage for desktop analysis results
- **Cross-Device Access**: Access strategies on any device

## Deployment

### Development
- Runs on `localhost:3000`
- Hot reloading enabled
- TypeScript type checking

### Production
- Built with `npm run build`
- Optimized for performance
- CDN-ready static assets
- Progressive Web App capabilities

## Contributing

### Code Style
- TypeScript strict mode enabled
- ESLint and Prettier for code formatting
- Conventional commit messages
- Component-driven development

### Testing
- Jest for unit testing
- React Testing Library for component testing
- Cypress for end-to-end testing

## Performance Optimizations

- **Code Splitting**: Automatic route-based code splitting
- **Image Optimization**: Next.js Image component with optimization
- **Bundle Analysis**: Built-in bundle analyzer
- **Caching**: Aggressive caching with React Query
- **SEO**: Meta tags and structured data

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## License

Proprietary - SpygateAI Pro Football Analysis Tool
