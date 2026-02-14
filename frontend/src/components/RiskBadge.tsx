import { motion } from 'framer-motion';

interface RiskBadgeProps {
  score: number;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
}

export default function RiskBadge({ score, size = 'md', showLabel = true }: RiskBadgeProps) {
  const sizes = {
    sm: { dimension: 48, stroke: 4, fontSize: 'text-sm' },
    md: { dimension: 64, stroke: 5, fontSize: 'text-lg' },
    lg: { dimension: 96, stroke: 6, fontSize: 'text-2xl' }
  };

  const config = sizes[size];
  const radius = (config.dimension / 2) - (config.stroke / 2);
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;

  const getColor = () => {
    if (score <= 30) return { stroke: '#10B981', fill: '#D1FAE5', text: 'text-green-600' };
    if (score <= 70) return { stroke: '#F59E0B', fill: '#FEF3C7', text: 'text-amber-600' };
    return { stroke: '#EF4444', fill: '#FEE2E2', text: 'text-red-600' };
  };

  const colors = getColor();

  return (
    <div className="flex flex-col items-center gap-1">
      <div style={{ width: config.dimension, height: config.dimension }} className="relative">
        <svg width={config.dimension} height={config.dimension} className="transform -rotate-90">
          {/* Background circle */}
          <circle
            cx={config.dimension / 2}
            cy={config.dimension / 2}
            r={radius}
            fill={colors.fill}
            stroke="#E5E7EB"
            strokeWidth={config.stroke}
          />
          {/* Animated progress circle */}
          <motion.circle
            cx={config.dimension / 2}
            cy={config.dimension / 2}
            r={radius}
            fill="none"
            stroke={colors.stroke}
            strokeWidth={config.stroke}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1, ease: "easeOut" }}
          />
        </svg>
        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span
            className={`font-bold font-mono ${colors.text} ${config.fontSize}`}
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            {score}
          </motion.span>
          {showLabel && (
            <span className="text-[10px] text-gray-500 font-medium uppercase tracking-wide">
              Risk
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
