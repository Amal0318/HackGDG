interface AlertAcknowledgmentProps {
  patientId: string;
  riskScore?: number;
  onAcknowledge: (note: string) => void;
  isAcknowledged: boolean;
}

export const AlertAcknowledgment = ({ 
  patientId, 
  riskScore, 
  onAcknowledge, 
  isAcknowledged 
}: AlertAcknowledgmentProps) => {
  const alertLevel = riskScore && riskScore > 0.85 ? 'CRITICAL' : 'WARNING';
  const alertColor = alertLevel === 'CRITICAL' ? 'red' : 'yellow';

  const handleClick = () => {
    if (isAcknowledged) {
      onAcknowledge(''); // Clear acknowledgment
    } else {
      const note = prompt('Add a note (optional):');
      if (note !== null) {
        onAcknowledge(note);
      }
    }
  };

  return (
    <div className={`bg-${alertColor}-50 border-l-4 border-${alertColor}-400 p-4 rounded-lg shadow-sm`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center">
          <svg className={`h-6 w-6 text-${alertColor}-400 mr-3`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <div>
            <h3 className={`text-sm font-semibold text-${alertColor}-800`}>
              {alertLevel} Alert - Patient {patientId}
            </h3>
            {riskScore !== undefined && (
              <p className={`text-xs text-${alertColor}-700 mt-1`}>
                Risk Score: {(riskScore * 100).toFixed(0)}%
              </p>
            )}
          </div>
        </div>
        <button
          onClick={handleClick}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            isAcknowledged
              ? 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              : `bg-${alertColor}-600 text-white hover:bg-${alertColor}-700`
          }`}
        >
          {isAcknowledged ? 'Acknowledged ✓' : 'Acknowledge Alert'}
        </button>
      </div>
      {isAcknowledged && (
        <div className="mt-2 text-xs text-gray-600">
          Alert acknowledged at {new Date().toLocaleTimeString()}
        </div>
      )}
    </div>
  );
}
