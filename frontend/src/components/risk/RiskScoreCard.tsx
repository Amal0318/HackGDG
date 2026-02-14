interface RiskScoreCardProps {
  riskScore: number;
  riskLevel?: string;
}

const getRiskColor = (score: number): string => {
  if (score < 0.4) return 'from-primary-400 to-primary-500';
  if (score < 0.7) return 'from-yellow-400 to-yellow-500';
  return 'from-red-400 to-red-500';
};

const getRiskBorderColor = (score: number): string => {
  if (score < 0.4) return 'border-primary-400';
  if (score < 0.7) return 'border-yellow-400';
  return 'border-red-400';
};

const getRiskTextColor = (score: number): string => {
  if (score < 0.4) return 'text-primary-600';
  if (score < 0.7) return 'text-yellow-600';
  return 'text-red-600';
};

const getRiskLabel = (score: number): string => {
  if (score < 0.4) return 'LOW';
  if (score < 0.7) return 'MODERATE';
  if (score < 0.85) return 'HIGH';
  return 'CRITICAL';
};

export const RiskScoreCard = ({ riskScore, riskLevel }: RiskScoreCardProps) => {
  const shouldPulse = riskScore > 0.8;
  const label = riskLevel || getRiskLabel(riskScore);

  return (
    <div 
      className={`rounded-lg border-2 p-6 bg-white shadow-sm ${getRiskBorderColor(riskScore)} ${
        shouldPulse ? 'animate-pulse-subtle' : ''
      } transition-all`}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-gray-900 font-semibold text-lg">Risk Score</h3>
        <span className={`text-sm font-bold px-3 py-1 rounded-full ${getRiskTextColor(riskScore)} bg-gray-50`}>
          {label}
        </span>
      </div>

      <div className="relative">
        {/* Score Display */}
        <div className="flex items-baseline space-x-2 mb-4">
          <span className={`text-5xl font-bold ${getRiskTextColor(riskScore)}`}>
            {(riskScore * 100).toFixed(0)}
          </span>
          <span className="text-gray-500 text-2xl">%</span>
        </div>

        {/* Risk Bar */}
        <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden">
          <div
            className={`h-full bg-gradient-to-r ${getRiskColor(riskScore)} transition-all duration-500 ease-out`}
            style={{ width: `${riskScore * 100}%` }}
          />
        </div>

        {/* Threshold markers */}
        <div className="relative h-6 mt-1">
          <div className="absolute left-[40%] top-0 w-px h-4 bg-gray-400" />
          <div className="absolute left-[70%] top-0 w-px h-4 bg-gray-400" />
          <div className="absolute left-[85%] top-0 w-px h-4 bg-gray-400" />
          
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>0</span>
            <span>40</span>
            <span>70</span>
            <span>85</span>
            <span>100</span>
          </div>
        </div>
      </div>

      <div className="mt-4 pt-4 border-t border-gray-200">
        <p className="text-gray-600 text-sm">
          {riskScore < 0.4 && 'Patient vitals are stable. Continue monitoring.'}
          {riskScore >= 0.4 && riskScore < 0.7 && 'Elevated risk detected. Monitor closely.'}
          {riskScore >= 0.7 && riskScore < 0.85 && 'High risk of deterioration. Consider intervention.'}
          {riskScore >= 0.85 && 'Critical risk level. Immediate intervention recommended.'}
        </p>
      </div>
    </div>
  );
};
