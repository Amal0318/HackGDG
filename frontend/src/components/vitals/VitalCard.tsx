interface VitalCardProps {
  title: string;
  value: number | string;
  unit: string;
  status?: 'normal' | 'warning' | 'critical';
  icon?: string;
}

export const VitalCard = ({ title, value, unit, status = 'normal' }: VitalCardProps) => {
  const getStatusColor = () => {
    switch (status) {
      case 'warning':
        return 'border-yellow-400 bg-yellow-50';
      case 'critical':
        return 'border-red-400 bg-red-50';
      default:
        return 'border-gray-200 bg-white';
    }
  };

  const getValueColor = () => {
    switch (status) {
      case 'warning':
        return 'text-yellow-600';
      case 'critical':
        return 'text-red-600';
      default:
        return 'text-gray-900';
    }
  };

  return (
    <div className={`rounded-lg border-2 p-4 shadow-sm ${getStatusColor()} transition-all`}>
      <div className="text-gray-600 text-sm font-medium mb-2">{title}</div>
      <div className="flex items-baseline space-x-2">
        <span className={`text-3xl font-bold ${getValueColor()}`}>
          {typeof value === 'number' ? value.toFixed(1) : value}
        </span>
        <span className="text-gray-500 text-lg">{unit}</span>
      </div>
    </div>
  );
};
