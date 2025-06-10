/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './src/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      // FACEIT-style SpygateAI color palette
      colors: {
        // Primary SpygateAI brand colors
        spygate: {
          orange: '#ff6b35',
          'orange-dark': '#e55a2b',
          'orange-light': '#ff8b55',
        },
        
        // Dark theme backgrounds (FACEIT-inspired)
        dark: {
          bg: '#0f0f0f',
          surface: '#1a1a1a',
          elevated: '#2a2a2a',
          border: '#333333',
          text: '#ffffff',
          'text-secondary': '#cccccc',
          'text-muted': '#888888',
        },
        
        // Status colors for performance tiers
        tier: {
          clutch: '#10b981',    // Green for clutch plays
          big: '#3b82f6',       // Blue for big plays
          good: '#8b5cf6',      // Purple for good plays
          average: '#f59e0b',   // Yellow for average
          poor: '#ef4444',      // Red for poor plays
          turnover: '#dc2626',  // Dark red for turnovers
        },
        
        // Game-specific colors
        game: {
          madden: '#1e40af',
          cfb: '#dc2626',
          universal: '#6366f1',
        },
        
        // Status indicators
        success: '#10b981',
        warning: '#f59e0b',
        error: '#ef4444',
        info: '#3b82f6',
      },
      
      // Typography scale
      fontSize: {
        'xs': ['0.75rem', { lineHeight: '1rem' }],
        'sm': ['0.875rem', { lineHeight: '1.25rem' }],
        'base': ['1rem', { lineHeight: '1.5rem' }],
        'lg': ['1.125rem', { lineHeight: '1.75rem' }],
        'xl': ['1.25rem', { lineHeight: '1.75rem' }],
        '2xl': ['1.5rem', { lineHeight: '2rem' }],
        '3xl': ['1.875rem', { lineHeight: '2.25rem' }],
        '4xl': ['2.25rem', { lineHeight: '2.5rem' }],
      },
      
      // Spacing for consistent layout
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '120': '30rem',
      },
      
      // Border radius for modern design
      borderRadius: {
        'xl': '0.75rem',
        '2xl': '1rem',
        '3xl': '1.5rem',
      },
      
      // Box shadows for depth
      boxShadow: {
        'spygate': '0 4px 6px -1px rgba(255, 107, 53, 0.1), 0 2px 4px -1px rgba(255, 107, 53, 0.06)',
        'dark': '0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2)',
      },
      
      // Animation for smooth interactions
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
      
      // Backdrop blur for modals
      backdropBlur: {
        'xs': '2px',
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
  darkMode: 'class',
}; 