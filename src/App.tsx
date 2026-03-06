import { useState, useEffect } from 'react';
import { useAppStore } from './store/appStore';
import { checkBackendHealth } from './services/api';
import HomeScreen from './components/home/HomeScreen';
import ProjectDashboard from './components/dashboard/ProjectDashboard';
import './index.css';

function App() {
  const { currentProject, darkMode } = useAppStore();
  const [backendReady, setBackendReady] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check backend health on startup
    const checkHealth = async () => {
      setLoading(true);
      let attempts = 0;
      const maxAttempts = 30; // 15 seconds max wait

      while (attempts < maxAttempts) {
        const healthy = await checkBackendHealth();
        if (healthy) {
          setBackendReady(true);
          setLoading(false);
          return;
        }
        await new Promise((r) => setTimeout(r, 500));
        attempts++;
      }

      setLoading(false);
    };

    checkHealth();
  }, []);

  // Apply dark mode class
  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

  if (loading) {
    return <LoadingScreen />;
  }

  if (!backendReady) {
    return <BackendErrorScreen onRetry={() => window.location.reload()} />;
  }

  return (
    <div className="app">
      {currentProject ? <ProjectDashboard /> : <HomeScreen />}
    </div>
  );
}

function LoadingScreen() {
  return (
    <div className="loading-screen">
      <div className="loading-content">
        <div className="logo-container">
          <div className="logo-glow"></div>
          <svg className="logo" viewBox="0 0 64 64" fill="none">
            <circle cx="32" cy="32" r="28" stroke="url(#gradient)" strokeWidth="2" />
            <circle cx="32" cy="32" r="18" stroke="url(#gradient)" strokeWidth="2" opacity="0.6" />
            <circle cx="32" cy="32" r="8" fill="url(#gradient)" />
            <defs>
              <linearGradient id="gradient" x1="0" y1="0" x2="64" y2="64">
                <stop offset="0%" stopColor="#6366f1" />
                <stop offset="50%" stopColor="#8b5cf6" />
                <stop offset="100%" stopColor="#d946ef" />
              </linearGradient>
            </defs>
          </svg>
        </div>
        <h1 className="loading-title">Singularity Vision</h1>
        <p className="loading-subtitle">Initializing...</p>
        <div className="loading-bar">
          <div className="loading-bar-progress"></div>
        </div>
      </div>
      <style>{`
        .loading-screen {
          width: 100vw;
          height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--color-bg-primary);
        }

        .loading-content {
          text-align: center;
        }

        .logo-container {
          position: relative;
          width: 120px;
          height: 120px;
          margin: 0 auto 32px;
        }

        .logo-glow {
          position: absolute;
          inset: -20px;
          background: rgba(99, 102, 241, 0.2);
          border-radius: 50%;
          filter: blur(30px);
          animation: pulse 2s ease infinite;
        }

        .logo {
          width: 100%;
          height: 100%;
          animation: spin 20s linear infinite;
        }

        .loading-title {
          font-size: 2rem;
          font-weight: 700;
          color: var(--color-accent-primary);
          margin-bottom: 8px;
        }

        .loading-subtitle {
          color: var(--color-text-muted);
          margin-bottom: 24px;
        }

        .loading-bar {
          width: 200px;
          height: 4px;
          background: var(--color-bg-tertiary);
          border-radius: 2px;
          margin: 0 auto;
          overflow: hidden;
        }

        .loading-bar-progress {
          width: 40%;
          height: 100%;
          background: var(--color-accent-primary);
          border-radius: 2px;
          animation: loadingSlide 1.5s ease infinite;
        }

        @keyframes loadingSlide {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(350%); }
        }
      `}</style>
    </div>
  );
}

function BackendErrorScreen({ onRetry }: { onRetry: () => void }) {
  return (
    <div className="error-screen">
      <div className="error-content">
        <div className="error-icon">⚠️</div>
        <h2>Backend Connection Failed</h2>
        <p>Could not connect to the Python backend. Please make sure:</p>
        <ul>
          <li>Python 3.8+ is installed</li>
          <li>Required dependencies are installed</li>
          <li>No other application is using port 8765</li>
        </ul>
        <button className="btn btn-primary btn-lg" onClick={onRetry}>
          Retry Connection
        </button>
      </div>
      <style>{`
        .error-screen {
          width: 100vw;
          height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          background: var(--color-bg-primary);
        }

        .error-content {
          text-align: center;
          max-width: 400px;
          padding: 48px;
          background: var(--color-bg-secondary);
          border: 1px solid var(--color-border);
          border-radius: 24px;
        }

        .error-icon {
          font-size: 48px;
          margin-bottom: 16px;
        }

        .error-content h2 {
          margin-bottom: 16px;
          color: var(--color-error);
        }

        .error-content p {
          margin-bottom: 16px;
        }

        .error-content ul {
          text-align: left;
          margin-bottom: 24px;
          padding-left: 24px;
        }

        .error-content li {
          color: var(--color-text-secondary);
          margin-bottom: 8px;
        }
      `}</style>
    </div>
  );
}

export default App;
