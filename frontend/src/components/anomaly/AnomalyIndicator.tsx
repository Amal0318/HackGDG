interface AnomalyIndicatorProps {
  hasAnomaly: boolean;
  anomalyType?: string;
}

export const AnomalyIndicator = ({ hasAnomaly, anomalyType }: AnomalyIndicatorProps) => {
  if (!hasAnomaly) {
    return null;
  }

  return (
    <div className="inline-flex items-center px-3 py-1.5 rounded-lg bg-orange-100 border border-orange-300 animate-pulse">
      <svg className="h-4 w-4 text-orange-600 mr-2" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
      </svg>
      <div>
        <span className="text-sm font-medium text-orange-800">Anomaly</span>
        {anomalyType && (
          <span className="text-xs text-orange-700 ml-1">({anomalyType})</span>
        )}
      </div>
    </div>
  );
};
