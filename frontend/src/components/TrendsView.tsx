import { useState } from 'react';
import { Activity, Heart, TrendingUp, Settings } from 'lucide-react';
import RiskTrendChart from './RiskTrendChart';
import VitalsTrendChart from './VitalsTrendChart';
import { usePollingPatientData } from '../hooks/usePollingPatientData';

interface TrendsViewProps {
  patientId: string;
  className?: string;
}

const VITAL_OPTIONS = [
  { key: 'heart_rate', label: 'Heart Rate', icon: Heart },
  { key: 'systolic_bp', label: 'Systolic BP', icon: Activity },
  { key: 'diastolic_bp', label: 'Diastolic BP', icon: Activity },
  { key: 'spo2', label: 'SpO₂', icon: TrendingUp },
  { key: 'respiratory_rate', label: 'Resp Rate', icon: Activity },
  { key: 'temperature', label: 'Temperature', icon: TrendingUp },
];

export default function TrendsView({ patientId, className = '' }: TrendsViewProps) {
  const [selectedVitals, setSelectedVitals] = useState<string[]>(['heart_rate', 'systolic_bp', 'spo2']);
  const [showVitalSelector, setShowVitalSelector] = useState(false);
  
  // Use HTTP polling to keep data fresh without WebSockets
  const {
    isPolling,
    getPatientRiskHistory,
    getPatientVitalsHistory,
  } = usePollingPatientData(patientId, 2000);

  const riskData = getPatientRiskHistory(patientId);
  const vitalsData = getPatientVitalsHistory(patientId);
  const isRiskLive = isPolling && riskData.length > 0;
  const isVitalsLive = isPolling && vitalsData.length > 0;

  const toggleVital = (vitalKey: string) => {
    setSelectedVitals(prev => {
      if (prev.includes(vitalKey)) {
        return prev.filter(v => v !== vitalKey);
      } else {
        return [...prev, vitalKey];
      }
    });
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900">
          Live Patient Trends
        </h3>
        <div className="flex items-center gap-3">
          {/* Connection Status */}
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1">
              <div className={`w-2 h-2 rounded-full ${isRiskLive ? 'bg-green-500' : 'bg-gray-400'}`} />
              <span className="text-gray-600">Risk</span>
            </div>
            <div className="flex items-center gap-1">
              <div className={`w-2 h-2 rounded-full ${isVitalsLive ? 'bg-green-500' : 'bg-gray-400'}`} />
              <span className="text-gray-600">Vitals</span>
            </div>
          </div>
          
          {/* Vitals Selector */}
          <div className="relative">
            <button
              onClick={() => setShowVitalSelector(!showVitalSelector)}
              className="flex items-center gap-1 px-3 py-1.5 text-xs bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            >
              <Settings className="w-3 h-3" />
              Vitals ({selectedVitals.length})
            </button>
            
            {showVitalSelector && (
              <div className="absolute top-full right-0 mt-1 bg-white border border-gray-200 rounded-lg shadow-lg z-10 min-w-48">
                <div className="p-3">
                  <div className="text-xs font-medium text-gray-700 mb-2">Select Vitals to Display</div>
                  <div className="space-y-1">
                    {VITAL_OPTIONS.map(({ key, label, icon: Icon }) => (
                      <label key={key} className="flex items-center gap-2 text-xs cursor-pointer hover:bg-gray-50 p-1 rounded">
                        <input
                          type="checkbox"
                          checked={selectedVitals.includes(key)}
                          onChange={() => toggleVital(key)}
                          className="text-blue-600 rounded"
                        />
                        <Icon className="w-3 h-3 text-gray-500" />
                        <span>{label}</span>
                      </label>
                    ))}
                  </div>
                  <div className="mt-2 pt-2 border-t border-gray-200">
                    <button
                      onClick={() => setShowVitalSelector(false)}
                      className="text-xs text-blue-600 hover:text-blue-700"
                    >
                      Close
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Trend Chart */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-red-500" />
            <h4 className="font-medium text-gray-900">Risk Score Trend</h4>
            {isRiskLive && (
              <div className="flex items-center gap-1 text-xs text-green-600 font-medium">
                <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
                Live
              </div>
            )}
          </div>
          <RiskTrendChart 
            data={riskData} 
            height={280}
            showGrid={true}
          />
          {riskData.length === 0 && (
            <div className="text-center text-gray-500 text-sm py-8">
              Waiting for risk predictions...
            </div>
          )}
        </div>

        {/* Vitals Trend Chart */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center gap-2 mb-4">
            <Activity className="w-5 h-5 text-blue-500" />
            <h4 className="font-medium text-gray-900">Vital Signs Trend</h4>
            {isVitalsLive && (
              <div className="flex items-center gap-1 text-xs text-green-600 font-medium">
                <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
                Live
              </div>
            )}
          </div>
          <VitalsTrendChart 
            data={vitalsData} 
            height={280}
            selectedVitals={selectedVitals}
            showGrid={true}
          />
          {vitalsData.length === 0 && (
            <div className="text-center text-gray-500 text-sm py-8">
              Waiting for vitals data...
            </div>
          )}
        </div>
      </div>

      {/* Data Summary */}
      {(riskData.length > 0 || vitalsData.length > 0) && (
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
            <div>
              <span className="text-gray-600">Risk Points:</span>
              <span className="font-medium ml-1">{riskData.length}</span>
            </div>
            <div>
              <span className="text-gray-600">Vitals Points:</span>
              <span className="font-medium ml-1">{vitalsData.length}</span>
            </div>
            <div>
              <span className="text-gray-600">Time Range:</span>
              <span className="font-medium ml-1">
                {Math.max(riskData.length, vitalsData.length) * 2}s
              </span>
            </div>
            <div>
              <span className="text-gray-600">Update Rate:</span>
              <span className="font-medium ml-1">~2s</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}