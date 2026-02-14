interface AnomalyIndicatorProps {
  hasAnomaly: boolean;
  anomalyType?: string;
}

export const AnomalyIndicator = ({ hasAnomaly, anomalyType }: AnomalyIndicatorProps) => {
  if (!hasAnomaly) return null;

  return (
    <div className="inline-flex items-center space-x-2 px-3 py-1 bg-orange-50 border-2 border-orange-400 rounded-lg animate-pulse-subtle">
      <div className="w-2 h-2 rounded-full bg-orange-500 animate-ping-slow" />
      <div className="flex flex-col">
        <span className="text-xs font-semibold text-orange-700">ANOMALY DETECTED</span>
        {anomalyType && (
          <span className="text-xs text-orange-600">{anomalyType.replace('_', ' ')}</span>
        )}
      </div>
    </div>
  );
};