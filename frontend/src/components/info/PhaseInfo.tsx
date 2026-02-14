import { useStore } from '../../store';

interface PhaseDetails {
  title: string;
  description: string;
  features: string[];
  actions: string[];
}

const phaseDetails: Record<string, PhaseDetails> = {
  phase1: {
    title: '📊 Phase 1: Real-Time Vital Signs Monitor',
    description: 'Foundation of the ICU Digital Twin - displays live vital signs with historical trend visualization.',
    features: [
      'Live vital signs display (Heart Rate, BP, SpO2, Temp, RR)',
      'Real-time data streaming via WebSocket',
      'Historical trend charts with Recharts',
      'Color-coded status indicators (normal/warning/critical)',
      'Patient list with state badges',
    ],
    actions: [
      'Select patients from sidebar',
      'View current vitals',
      'Monitor trends over time',
    ],
  },
  phase2: {
    title: '⚠️ Phase 2: Risk Assessment & Alerts',
    description: 'ML-powered risk scoring integrated with the monitoring system to predict patient deterioration.',
    features: [
      'Real-time risk score calculation (0-100)',
      'Risk level classification (Low/Medium/High/Critical)',
      'Risk trend visualization over time',
      'Automated alert banners for high-risk patients',
      'Color-coded risk indicators',
    ],
    actions: [
      'Monitor risk scores',
      'View risk trends',
      'Respond to automated alerts',
      'Track risk level changes',
    ],
  },
  phase3: {
    title: '🔍 Phase 3: Anomaly Detection Engine',
    description: 'Real-time anomaly detection powered by Pathway streaming engine for early warning of clinical events.',
    features: [
      'Real-time anomaly detection (Hypotension, Tachycardia, Hypoxia, Sepsis)',
      'Event timeline with chronological event tracking',
      'Anomaly type classification and badges',
      'Integration with Pathway streaming engine',
      'Combined risk + anomaly alerting',
    ],
    actions: [
      'View anomaly indicators',
      'Browse event timeline',
      'Click events for details',
      'Monitor multiple event types',
    ],
  },
  phase4: {
    title: '⚡ Phase 4: Interactive Clinical Workflows',
    description: 'Clinician-facing interactive features for alert management, data analysis, and workflow optimization.',
    features: [
      'Alert acknowledgment system with notes',
      'CSV/JSON data export functionality',
      'Advanced event filtering (type, state, date range)',
      'Anomaly details modal with full context',
      'Acknowledged alert tracking',
    ],
    actions: [
      'Acknowledge alerts with notes',
      'Export patient data (CSV/JSON)',
      'Filter events by criteria',
      'View detailed anomaly information',
      'Clear acknowledgments',
    ],
  },
  all: {
    title: '🎯 Complete ICU Digital Twin System',
    description: 'All 4 phases integrated - full-featured patient monitoring with ML risk assessment, anomaly detection, and clinical workflows.',
    features: [
      'All Phase 1-4 features combined',
      'Real-time vital signs monitoring',
      'ML-powered risk assessment',
      'Anomaly detection and alerting',
      'Interactive clinical workflows',
      'Complete patient insight dashboard',
    ],
    actions: [
      'All actions from Phases 1-4',
      'Comprehensive patient monitoring',
      'End-to-end clinical decision support',
    ],
  },
};

export const PhaseInfo = () => {
  const selectedPhase = useStore(state => state.selectedPhase);

  if (selectedPhase === 'phase5') {
    return null; // Multi-patient dashboard doesn't use this
  }

  const phase = phaseDetails[selectedPhase];

  if (!phase) return null;

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 mb-6">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <h3 className="text-lg font-bold text-gray-900 mb-2">{phase.title}</h3>
          <p className="text-sm text-gray-600 mb-4">{phase.description}</p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="text-sm font-semibold text-gray-900 mb-2">✨ Features</h4>
              <ul className="space-y-1">
                {phase.features.map((feature, idx) => (
                  <li key={idx} className="text-xs text-gray-700 flex items-start">
                    <span className="text-primary-500 mr-2">•</span>
                    <span>{feature}</span>
                  </li>
                ))}
              </ul>
            </div>
            
            <div>
              <h4 className="text-sm font-semibold text-gray-900 mb-2">🎯 Available Actions</h4>
              <ul className="space-y-1">
                {phase.actions.map((action, idx) => (
                  <li key={idx} className="text-xs text-gray-700 flex items-start">
                    <span className="text-primary-500 mr-2">→</span>
                    <span>{action}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
        
        <div className="ml-4">
          <div className="bg-primary-50 rounded-lg px-3 py-1">
            <span className="text-xs font-medium text-primary-700">
              {selectedPhase === 'all' ? 'All Phases' : selectedPhase.toUpperCase()}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
