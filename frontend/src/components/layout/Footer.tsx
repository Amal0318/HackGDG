import { useStore } from '../../store';

export const Footer = () => {
  const wsConnected = useStore(state => state.wsConnected);
  const healthStatus = useStore(state => state.healthStatus);
  const viewMode = useStore(state => state.viewMode);
  const selectedPhase = useStore(state => state.selectedPhase);

  const getServiceStatus = (status: string | undefined) => {
    if (!status) return { color: 'bg-gray-400', text: 'Unknown' };
    
    const statusLower = status.toLowerCase();
    if (statusLower.includes('connected') || statusLower.includes('healthy')) {
      return { color: 'bg-primary-500', text: status };
    }
    if (statusLower.includes('initializing') || statusLower.includes('starting')) {
      return { color: 'bg-yellow-500', text: status };
    }
    return { color: 'bg-red-500', text: status };
  };

  return (
    <footer className="bg-white border-t border-gray-200 px-6 py-3 shadow-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-6">
          <div className="flex items-center space-x-2">
            <span className="text-gray-600 text-sm">WebSocket:</span>
            <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-primary-500' : 'bg-red-500'}`}></div>
            <span className="text-gray-900 text-sm font-medium">
              {wsConnected ? 'Active' : 'Disconnected'}
            </span>
          </div>

          {healthStatus && (
            <>
              <div className="flex items-center space-x-2">
                <span className="text-gray-600 text-sm">Kafka:</span>
                <div className={`w-2 h-2 rounded-full ${getServiceStatus(healthStatus.kafka).color}`}></div>
                <span className="text-gray-900 text-sm font-medium">
                  {getServiceStatus(healthStatus.kafka).text}
                </span>
              </div>

              <div className="flex items-center space-x-2">
                <span className="text-gray-600 text-sm">Pathway:</span>
                <div className={`w-2 h-2 rounded-full ${getServiceStatus(healthStatus.pathway).color}`}></div>
                <span className="text-gray-900 text-sm font-medium">
                  {getServiceStatus(healthStatus.pathway).text}
                </span>
              </div>

              <div className="flex items-center space-x-2">
                <span className="text-gray-600 text-sm">ML Service:</span>
                <div className={`w-2 h-2 rounded-full ${getServiceStatus(healthStatus.ml_service).color}`}></div>
                <span className="text-gray-900 text-sm font-medium">
                  {getServiceStatus(healthStatus.ml_service).text}
                </span>
              </div>
            </>
          )}
        </div>

        <div className="text-gray-500 text-xs">
          {viewMode === 'multi-patient' && 'Phase 5 - Multi-Patient Monitoring Dashboard'}
          {viewMode === 'single-patient' && selectedPhase === 'all' && '🎯 All Phases - Complete ICU Digital Twin'}
          {viewMode === 'single-patient' && selectedPhase === 'phase1' && '📊 Phase 1 - Real-Time Vital Signs Monitor'}
          {viewMode === 'single-patient' && selectedPhase === 'phase2' && '⚠️ Phase 2 - Risk Assessment & Alerts'}
          {viewMode === 'single-patient' && selectedPhase === 'phase3' && '🔍 Phase 3 - Anomaly Detection Engine'}
          {viewMode === 'single-patient' && selectedPhase === 'phase4' && '⚡ Phase 4 - Interactive Clinical Workflows'}
        </div>
      </div>
    </footer>
  );
};
