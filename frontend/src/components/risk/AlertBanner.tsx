interface AlertBannerProps {
  riskScore: number;
  patientId: string;
}

export const AlertBanner = ({ riskScore, patientId }: AlertBannerProps) => {
  if (riskScore < 0.7) return null;

  const isCritical = riskScore >= 0.85;

  return (
    <div
      className={`rounded-lg border-2 p-4 mb-6 ${
        isCritical
          ? 'bg-red-900/20 border-red-500 animate-pulse-subtle'
          : 'bg-yellow-900/20 border-yellow-500'
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className={`w-3 h-3 rounded-full ${isCritical ? 'bg-red-500' : 'bg-yellow-500'} animate-ping-slow`} />
          <div>
            <h3 className={`font-bold text-lg ${isCritical ? 'text-red-600' : 'text-yellow-600'}`}>
              {isCritical ? '🚨 CRITICAL ALERT' : '⚠️ WARNING'}
            </h3>
            <p className="text-gray-900 text-sm mt-1">
              Patient {patientId} - Risk score: {(riskScore * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        <div className="text-right">
          <p className={`font-semibold ${isCritical ? 'text-red-600' : 'text-yellow-600'}`}>
            {isCritical ? 'Immediate Intervention Required' : 'Close Monitoring Advised'}
          </p>
          <p className="text-gray-700 text-xs mt-1">
            {isCritical 
              ? 'Contact medical team immediately'
              : 'Assess patient status and prepare for possible intervention'
            }
          </p>
        </div>
      </div>
    </div>
  );
};
