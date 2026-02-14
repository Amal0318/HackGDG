import { useEffect } from 'react';
import { Header } from './components/layout/Header';
import { Sidebar } from './components/layout/Sidebar';
import { MainPanel } from './components/layout/MainPanel';
import { MultiPatientDashboard } from './components/dashboard/MultiPatientDashboard';
import { Footer } from './components/layout/Footer';
import { useVitalStream } from './hooks/useVitalStream';
import { useHealthStatus } from './hooks/useHealthStatus';
import { useStore } from './store';

function App() {
  // Initialize WebSocket connection
  useVitalStream();
  
  // Initialize health status polling
  useHealthStatus();
  
  const viewMode = useStore(state => state.viewMode);
  const setViewMode = useStore(state => state.setViewMode);
  
  // Handle navigation from multi-patient to single-patient view
  useEffect(() => {
    const handleNavigateToPatient = () => {
      setViewMode('single-patient');
    };
    
    window.addEventListener('navigate-to-patient', handleNavigateToPatient);
    return () => window.removeEventListener('navigate-to-patient', handleNavigateToPatient);
  }, [setViewMode]);

  return (
    <div className="flex flex-col h-screen bg-background-light">
      <Header />
      
      <div className="flex flex-1 overflow-hidden">
        {viewMode === 'single-patient' && <Sidebar />}
        {viewMode === 'single-patient' ? <MainPanel /> : <MultiPatientDashboard />}
      </div>
      
      <Footer />
    </div>
  );
}

export default App;
