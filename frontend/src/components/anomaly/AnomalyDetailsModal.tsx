import { VitalMessage } from '../../types';

interface AnomalyDetailsModalProps {
  event: VitalMessage | null;
  isOpen: boolean;
  onClose: () => void;
}

export const AnomalyDetailsModal = ({ event, isOpen, onClose }: AnomalyDetailsModalProps) => {
  if (!isOpen || !event) return null;

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('en-US', {
      dateStyle: 'medium',
      timeStyle: 'medium'
    });
  };

  const getEventSeverity = () => {
    if (event.state === 'CRITICAL') return { color: 'red', text: 'Critical State' };
    if (event.state === 'INTERVENTION') return { color: 'red', text: 'Intervention Required' };
    if (event.state === 'EARLY_DETERIORATION') return { color: 'yellow', text: 'Early Deterioration' };
    if (event.anomaly_detected) return { color: 'orange', text: 'Anomaly Detected' };
    return { color: 'teal', text: 'Stable Event' };
  };

  const severity = getEventSeverity();

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 p-4">
      <div className="bg-white rounded-lg shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className={`bg-${severity.color}-50 border-b-2 border-${severity.color}-400 px-6 py-4`}>
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-gray-900 text-2xl font-bold">{severity.text}</h2>
              <p className="text-gray-600 text-sm mt-1">
                Patient: {event.patient_id} | {formatTimestamp(event.timestamp)}
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-gray-500 hover:text-gray-900 text-3xl font-bold leading-none"
              aria-label="Close modal"
            >
              ×
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="px-6 py-6 space-y-6">
          {/* Event Summary */}
          <div>
            <h3 className="text-gray-900 text-lg font-semibold mb-3">Event Summary</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-gray-50 rounded-lg p-4">
                <p className="text-gray-600 text-sm">Patient State</p>
                <p className={`text-lg font-bold mt-1 ${
                  event.state === 'CRITICAL' || event.state === 'INTERVENTION' ? 'text-red-600' :
                  event.state === 'EARLY_DETERIORATION' ? 'text-yellow-600' :
                  'text-primary-600'
                }`}>
                  {event.state}
                </p>
              </div>
              <div className="bg-gray-50 rounded-lg p-4">
                <p className="text-gray-600 text-sm">Event Type</p>
                <p className="text-lg font-bold mt-1 text-gray-900">{event.event_type}</p>
              </div>
              {event.risk_score !== undefined && (
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-gray-600 text-sm">Risk Score</p>
                  <p className={`text-lg font-bold mt-1 ${
                    event.risk_score >= 70 ? 'text-red-600' :
                    event.risk_score >= 50 ? 'text-yellow-600' :
                    'text-primary-600'
                  }`}>
                    {event.risk_score.toFixed(1)}%
                  </p>
                </div>
              )}
              {event.risk_level && (
                <div className="bg-gray-50 rounded-lg p-4">
                  <p className="text-gray-600 text-sm">Risk Level</p>
                  <p className="text-lg font-bold mt-1 text-gray-900">{event.risk_level}</p>
                </div>
              )}
            </div>
          </div>

          {/* Anomaly Details */}
          {event.anomaly_detected && (
            <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
              <h3 className="text-orange-900 text-lg font-semibold mb-2 flex items-center">
                <span className="mr-2">🔍</span>
                Anomaly Analysis
              </h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-orange-800 font-medium">Type:</span>
                  <span className="text-orange-900 font-bold bg-orange-100 px-3 py-1 rounded">
                    {event.anomaly_type || 'UNKNOWN'}
                  </span>
                </div>
                <p className="text-orange-800 text-sm mt-2">
                  This anomaly was detected by the Pathway Intelligence engine based on vital sign patterns
                  that deviate from expected norms.
                </p>
              </div>
            </div>
          )}

          {/* Vital Signs */}
          <div>
            <h3 className="text-gray-900 text-lg font-semibold mb-3">Vital Signs at Event Time</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              <VitalBox
                label="Heart Rate"
                value={event.heart_rate}
                unit="bpm"
                normalRange="60-100"
              />
              <VitalBox
                label="SpO2"
                value={event.spo2}
                unit="%"
                normalRange="95-100"
              />
              <VitalBox
                label="Systolic BP"
                value={event.systolic_bp}
                unit="mmHg"
                normalRange="90-120"
              />
              <VitalBox
                label="Diastolic BP"
                value={event.diastolic_bp}
                unit="mmHg"
                normalRange="60-80"
              />
              <VitalBox
                label="Resp. Rate"
                value={event.respiratory_rate}
                unit="bpm"
                normalRange="12-20"
              />
              <VitalBox
                label="Temperature"
                value={event.temperature}
                unit="°C"
                normalRange="36.5-37.5"
              />
              <VitalBox
                label="Shock Index"
                value={event.shock_index}
                unit=""
                normalRange="0.5-0.7"
              />
            </div>
          </div>

          {/* Clinical Recommendations */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="text-blue-900 text-lg font-semibold mb-2">💡 Clinical Considerations</h3>
            <ul className="text-blue-800 text-sm space-y-2 list-disc list-inside">
              {(event.state === 'CRITICAL' || event.state === 'INTERVENTION') && (
                <>
                  <li>Immediate clinician assessment required</li>
                  <li>Prepare for potential intervention</li>
                  <li>Verify vital sign accuracy</li>
                </>
              )}
              {event.state === 'EARLY_DETERIORATION' && (
                <>
                  <li>Increased monitoring frequency recommended</li>
                  <li>Review recent medications and interventions</li>
                  <li>Consider trending vital signs</li>
                </>
              )}
              {event.event_type === 'SEPSIS_ALERT' && (
                <>
                  <li>Consider sepsis screening protocol</li>
                  <li>Review recent lab results</li>
                  <li>Monitor for additional signs of infection</li>
                </>
              )}
              {event.event_type === 'TACHYCARDIA' && (
                <>
                  <li>Review cardiac medications</li>
                  <li>Check for vagal stimulation causes</li>
                  <li>Consider ECG if persistent</li>
                </>
              )}
              {event.event_type === 'HYPOTENSION' && (
                <>
                  <li>Assess fluid status and perfusion</li>
                  <li>Review blood pressure medications</li>
                  <li>Monitor urine output</li>
                </>
              )}
              {event.event_type === 'HYPOXIA' && (
                <>
                  <li>Check airway and breathing</li>
                  <li>Increase oxygen supplementation if needed</li>
                  <li>Consider respiratory support</li>
                </>
              )}
            </ul>
          </div>
        </div>

        {/* Footer */}
        <div className="bg-gray-50 px-6 py-4 border-t border-gray-200">
          <button
            onClick={onClose}
            className="bg-primary-500 hover:bg-primary-600 text-white px-6 py-2 rounded-lg font-medium transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

// Helper component for vital sign boxes
const VitalBox = ({ label, value, unit, normalRange }: {
  label: string;
  value: number;
  unit: string;
  normalRange: string;
}) => (
  <div className="bg-white border border-gray-200 rounded-lg p-3">
    <p className="text-gray-600 text-xs">{label}</p>
    <p className="text-gray-900 text-xl font-bold mt-1">
      {value.toFixed(1)} <span className="text-sm font-normal text-gray-600">{unit}</span>
    </p>
    <p className="text-gray-500 text-xs mt-1">Normal: {normalRange}</p>
  </div>
);
