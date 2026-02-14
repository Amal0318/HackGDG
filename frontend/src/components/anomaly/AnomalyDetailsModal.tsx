import { VitalMessage } from '../../types';

interface AnomalyDetailsModalProps {
  event: VitalMessage | null;
  isOpen: boolean;
  onClose: () => void;
}

export const AnomalyDetailsModal = ({ event, isOpen, onClose }: AnomalyDetailsModalProps) => {
  if (!isOpen || !event) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
        <div className="p-6">
          {/* Header */}
          <div className="flex justify-between items-start mb-6">
            <div>
              <h2 className="text-2xl font-bold text-gray-900">Event Details</h2>
              <p className="text-sm text-gray-500 mt-1">
                Patient: {event.patient_id}
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Event Information */}
          <div className="space-y-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">Event Type</h3>
              <p className="text-lg font-medium text-gray-900">{event.event_type}</p>
            </div>

            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">Patient State</h3>
              <p className="text-lg font-medium text-gray-900">{event.state}</p>
            </div>

            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-sm font-semibold text-gray-700 mb-2">Timestamp</h3>
              <p className="text-lg font-medium text-gray-900">
                {new Date(event.timestamp).toLocaleString()}
              </p>
            </div>

            {/* Vital Signs */}
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Vital Signs</h3>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-xs text-gray-500">Heart Rate</p>
                  <p className="text-sm font-medium text-gray-900">{event.heart_rate} bpm</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Blood Pressure</p>
                  <p className="text-sm font-medium text-gray-900">{event.systolic_bp}/{event.diastolic_bp} mmHg</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">SpO2</p>
                  <p className="text-sm font-medium text-gray-900">{event.spo2}%</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Respiratory Rate</p>
                  <p className="text-sm font-medium text-gray-900">{event.respiratory_rate} /min</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Temperature</p>
                  <p className="text-sm font-medium text-gray-900">{event.temperature}°C</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500">Shock Index</p>
                  <p className="text-sm font-medium text-gray-900">{event.shock_index.toFixed(2)}</p>
                </div>
              </div>
            </div>

            {/* Risk Score */}
            {event.risk_score !== undefined && (
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-sm font-semibold text-gray-700 mb-2">Risk Assessment</h3>
                <div className="flex items-center space-x-4">
                  <div>
                    <p className="text-xs text-gray-500">Risk Score</p>
                    <p className="text-lg font-medium text-gray-900">{event.risk_score.toFixed(2)}</p>
                  </div>
                  {event.risk_level && (
                    <div>
                      <p className="text-xs text-gray-500">Risk Level</p>
                      <p className="text-lg font-medium text-gray-900">{event.risk_level}</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Anomaly Information */}
            {event.anomaly_detected && (
              <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded-lg">
                <h3 className="text-sm font-semibold text-yellow-800 mb-2">Anomaly Detected</h3>
                {event.anomaly_type && (
                  <p className="text-sm text-yellow-700">Type: {event.anomaly_type}</p>
                )}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="mt-6 flex justify-end">
            <button
              onClick={onClose}
              className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
