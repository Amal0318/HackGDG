interface VitalHighlightProps {
  title: string;
  value: number | string;
  unit: string;
  isHighlighted: boolean;
  status?: 'normal' | 'warning' | 'critical';
}

export const VitalHighlight = ({ title, value, unit, isHighlighted, status = 'normal' }: VitalHighlightProps) => {
  const getStatusColor = () => {
    if (!isHighlighted) {
      return status === 'critical' ? 'border-red-400 bg-red-50' :
             status === 'warning' ? 'border-yellow-400 bg-yellow-50' :
             'border-gray-200 bg-white';
    }
    return 'border-orange-400 bg-orange-50 ring-2 ring-orange-300';
  };

  const getValueColor = () => {
    if (isHighlighted) return 'text-orange-700';
    return status === 'critical' ? 'text-red-600' :
           status === 'warning' ? 'text-yellow-600' :
           'text-gray-900';
  };

  return (
    <div className={`rounded-lg border-2 p-4 shadow-sm ${getStatusColor()} transition-all ${isHighlighted ? 'animate-pulse-subtle' : ''}`}>
      <div className=\"flex items-center justify-between mb-2\">
        <div className=\"text-gray-600 text-sm font-medium\">{title}</div>
        {isHighlighted && (
          <span className=\"text-xs px-2 py-1 bg-orange-500 text-white rounded-full font-bold\">
            ANOMALY
          </span>
        )}
      </div>
      <div className=\"flex items-baseline space-x-2\">
        <span className={`text-3xl font-bold ${getValueColor()}`}>
          {typeof value === 'number' ? value.toFixed(1) : value}
        </span>
        <span className=\"text-gray-500 text-lg\">{unit}</span>
      </div>
    </div>
  );
};