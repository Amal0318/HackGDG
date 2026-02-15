import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { LineChart, Line, ResponsiveContainer } from 'recharts';
import { motion } from 'framer-motion';
import clsx from 'clsx';

interface VitalSignProps {
  label: string;
  value: number | string;
  unit: string;
  trend?: 'up' | 'down' | 'stable';
  status?: 'normal' | 'warning' | 'critical';
  trendData?: number[];
  className?: string;
}

export default function VitalSign({
  label,
  value,
  unit,
  trend = 'stable',
  status = 'normal',
  trendData,
  className
}: VitalSignProps) {
  const statusColors = {
    normal: 'text-gray-800',
    warning: 'text-amber-500',
    critical: 'text-red-500'
  };

  const trendIcons = {
    up: TrendingUp,
    down: TrendingDown,
    stable: Minus
  };

  const trendColors = {
    up: 'text-red-500',
    down: 'text-blue-500',
    stable: 'text-gray-400'
  };

  const TrendIcon = trendIcons[trend];

  const chartData = trendData?.slice(-10).map((val, idx) => ({ value: val, index: idx })) || [];

  return (
    <div className={clsx('flex flex-col gap-1', className)}>
      <span className="text-[10px] text-gray-500 font-medium uppercase tracking-wide">
        {label}
      </span>
      <div className="flex items-baseline gap-2">
        <motion.span 
          key={value}
          initial={{ scale: 1.2, color: '#3B82F6' }}
          animate={{ scale: 1, color: 'inherit' }}
          transition={{ duration: 0.3 }}
          className={clsx(
            'text-2xl font-bold font-mono transition-colors duration-300',
            statusColors[status],
            status !== 'normal' && 'animate-pulse'
          )}
        >
          {value}
        </motion.span>
        <span className="text-xs text-gray-500 font-medium">
          {unit}
        </span>
        <TrendIcon className={clsx('w-4 h-4 ml-auto', trendColors[trend])} />
      </div>
      {trendData && trendData.length > 0 && (
        <div className="h-6 -mx-1">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <Line
                type="monotone"
                dataKey="value"
                stroke={status === 'critical' ? '#EF4444' : status === 'warning' ? '#F59E0B' : '#10B981'}
                strokeWidth={1.5}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
