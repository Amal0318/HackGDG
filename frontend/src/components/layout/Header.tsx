import { useStore } from '../../store';

export const Header = () => {
  const healthStatus = useStore(state => state.healthStatus);
  const wsConnected = useStore(state => state.wsConnected);
  const viewMode = useStore(state => state.viewMode);
  const setViewMode = useStore(state => state.setViewMode);
  const selectedPhase = useStore(state => state.selectedPhase);
  const setSelectedPhase = useStore(state => state.setSelectedPhase);

  const now = new Date().toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  });

  const phases = [
    { id: 'all' as const, label: '🎯 All Phases', description: 'Complete System' },
    { id: 'phase1' as const, label: '📊 Phase 1', description: 'Real-Time Vitals' },
    { id: 'phase2' as const, label: '⚠️ Phase 2', description: 'Risk Assessment' },
    { id: 'phase3' as const, label: '🔍 Phase 3', description: 'Anomaly Detection' },
    { id: 'phase4' as const, label: '⚡ Phase 4', description: 'Interactivity' },
    { id: 'phase5' as const, label: '👥 Phase 5', description: 'Multi-Patient' },
  ];

  return (
    <header className="bg-white border-b border-gray-200 px-6 py-3 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-primary-500 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-xl">V</span>
          </div>
          <div>
            <h1 className="text-gray-900 text-2xl font-bold">VitalX</h1>
            <p className="text-gray-600 text-sm">ICU Digital Twin System</p>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          {/* View Mode Switcher */}
          <div className="flex items-center bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setViewMode('multi-patient')}
              className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                viewMode === 'multi-patient'
                  ? 'bg-primary-500 text-white shadow-sm'
                  : 'text-gray-700 hover:text-gray-900'
              }`}
            >
              👥 Dashboard
            </button>
            <button
              onClick={() => setViewMode('single-patient')}
              className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                viewMode === 'single-patient'
                  ? 'bg-primary-500 text-white shadow-sm'
                  : 'text-gray-700 hover:text-gray-900'
              }`}
            >
              👤 Single Patient
            </button>
          </div>
          
          <div className="flex items-center space-x-2 border-l border-gray-300 pl-4">
            <div className={`w-3 h-3 rounded-full ${wsConnected ? 'bg-primary-500' : 'bg-red-500'}`}></div>
            <span className="text-gray-700 text-sm font-medium">
              {wsConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>

          <div className="text-gray-600 text-sm border-l border-gray-300 pl-4">
            {now}
          </div>
        </div>
      </div>

      {/* Phase Navigation Tabs */}
      {viewMode === 'single-patient' && (
        <div className="flex items-center space-x-1 overflow-x-auto pb-1">
          {phases.map((phase) => (
            <button
              key={phase.id}
              onClick={() => {
                setSelectedPhase(phase.id);
                if (phase.id === 'phase5') {
                  setViewMode('multi-patient');
                }
              }}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all whitespace-nowrap ${
                selectedPhase === phase.id
                  ? 'bg-primary-500 text-white shadow'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
              title={phase.description}
            >
              {phase.label}
            </button>
          ))}
        </div>
      )}
    </header>
  );
};
