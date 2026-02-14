import { VitalMessage, PatientState } from '../../types';

interface PatientGridCardProps {
  patientId: string;
  latestVitals: VitalMessage | null;
  onSelect: () => void;
  isSelected: boolean;
}

export const PatientGridCard = ({ patientId, latestVitals, onSelect, isSelected }: PatientGridCardProps) => {
  if (!latestVitals) {
    return (
      <button
        onClick={onSelect}
        className={`bg-white border-2 rounded-lg p-4 transition-all hover:shadow-md ${
          isSelected ? 'border-primary-500 ring-2 ring-primary-200' : 'border-gray-200'
        }`}
      >
        <div className="text-center text-gray-500">
          <p className="font-bold text-lg">{patientId}</p>
          <p className="text-sm mt-2">No data</p>
        </div>
      </button>
    );
  }

  const getStateColor = (state: PatientState) => {
    switch (state) {
      case 'CRITICAL':
      case 'INTERVENTION':
        return 'bg-red-50 border-red-400';
      case 'EARLY_DETERIORATION':
        return 'bg-yellow-50 border-yellow-400';
      case 'STABLE':
        return 'bg-green-50 border-green-400';
      default:
        return 'bg-gray-50 border-gray-300';
    }
  };

  const getStateTextColor = (state: PatientState) => {
    switch (state) {
      case 'CRITICAL':
      case 'INTERVENTION':
        return 'text-red-700';
      case 'EARLY_DETERIORATION':
        return 'text-yellow-700';
      case 'STABLE':
        return 'text-green-700';
      default:
        return 'text-gray-700';
    }
  };

  const getRiskColor = (score?: number) => {
    if (!score) return 'text-gray-600';
    if (score >= 70) return 'text-red-600';
    if (score >= 50) return 'text-yellow-600';
    return 'text-primary-600';
  };

  return (
    <button
      onClick={onSelect}
      className={`${getStateColor(latestVitals.state)} border-2 rounded-lg p-4 transition-all hover:shadow-lg hover:scale-[1.02] text-left w-full ${
        isSelected ? 'ring-4 ring-primary-300' : ''
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-gray-900 font-bold text-lg">{patientId}</h3>
        <div className="flex items-center space-x-2">
          {latestVitals.anomaly_detected && (
            <span className="bg-orange-500 text-white text-xs px-2 py-1 rounded-full animate-pulse">
              ⚠️ Anomaly
            </span>
          )}
          <span className={`${getStateTextColor(latestVitals.state)} font-bold text-sm`}>
            {latestVitals.state}
          </span>
        </div>
      </div>

      {/* Quick Vitals */}
      <div className="grid grid-cols-2 gap-2 mb-3">
        <VitalQuick label="HR" value={latestVitals.heart_rate} unit="bpm" />
        <VitalQuick label="SpO2" value={latestVitals.spo2} unit="%" />
        <VitalQuick label="SBP" value={latestVitals.systolic_bp} unit="mmHg" />
        <VitalQuick label="DBP" value={latestVitals.diastolic_bp} unit="mmHg" />
      </div>

      {/* Risk Score */}
      {latestVitals.risk_score !== undefined && (
        <div className="bg-white bg-opacity-60 rounded px-3 py-2 flex items-center justify-between">
          <span className="text-gray-700 text-sm font-medium">Risk Score</span>
          <span className={`font-bold text-lg ${getRiskColor(latestVitals.risk_score)}`}>
            {latestVitals.risk_score.toFixed(0)}%
          </span>
        </div>
      )}

      {/* Last Update */}
      <p className="text-gray-600 text-xs mt-2">
        Updated: {new Date(latestVitals.timestamp).toLocaleTimeString()}
      </p>
    </button>
  );
};

// Helper component for compact vital display
const VitalQuick = ({ label, value, unit }: { label: string; value: number; unit: string }) => (
  <div className="bg-white bg-opacity-60 rounded px-2 py-1">
    <p className="text-gray-600 text-xs">{label}</p>
    <p className="text-gray-900 font-bold text-sm">
      {value.toFixed(0)} <span className="text-xs font-normal">{unit}</span>
    </p>
  </div>
);
