import { useEffect } from 'react';
import { Header } from './components/layout/Header';
import { Footer } from './components/layout/Footer';
import { Sidebar } from './components/layout/Sidebar';
import { MainPanel } from './components/layout/MainPanel';
import { MultiPatientDashboard } from './components/dashboard/MultiPatientDashboard';
import { SystemMonitoringPanel } from './components/monitoring/SystemMonitoringPanel';
import { useStore } from './store';
import { useHealthStatus } from './hooks/useHealthStatus';
import { useVitalStream } from './hooks/useVitalStream';

function App() {
  const { viewMode } = useStore();
  
  // Initialize health monitoring
  useHealthStatus();
  
  // Initialize vital stream
  useVitalStream();

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <Header />
      <div className="flex flex-1 overflow-hidden">
        {viewMode === 'single-patient' && <Sidebar />}
        <main className="flex-1 overflow-auto">
          {viewMode === 'multi-patient' ? (
            <MultiPatientDashboard />
          ) : (
            <MainPanel />
          )}
        </main>
      </div>
      <Footer />
    </div>
  );
}

export default App;
