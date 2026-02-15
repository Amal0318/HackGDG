import { motion } from 'framer-motion';

interface RiskBadgeProps {
  score?: number;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
}

export default function RiskBadge({ score = 0, size = 'md', showLabel = true }: RiskBadgeProps) {
  const getRiskLevel = () => {
    if (score <= 30) return { 
      label: 'LOW', 
      bgColor: 'bg-green-100', 
      textColor: 'text-green-700',
      dotColor: 'bg-green-500'
    };
    if (score <= 60) return { 
      label: 'MEDIUM', 
      bgColor: 'bg-yellow-100', 
      textColor: 'text-yellow-700',
      dotColor: 'bg-yellow-500'
    };
    if (score <= 80) return { 
      label: 'HIGH', 
      bgColor: 'bg-orange-100', 
      textColor: 'text-orange-700',
      dotColor: 'bg-orange-500'
    };
    return { 
      label: 'CRITICAL', 
      bgColor: 'bg-red-100', 
      textColor: 'text-red-700',
      dotColor: 'bg-red-500'
    };
  };

  const risk = getRiskLevel();
  
  // Format score - show as whole number percentage
  const formatScore = (score: number) => {
    return Math.round(score).toString();
  };

  // Size classes
  const sizeClasses = {
    sm: 'px-2 py-1 text-xs',
    md: 'px-3 py-1.5 text-xs',
    lg: 'px-4 py-2 text-sm'
  };

  const dotSizeClasses = {
    sm: 'w-1.5 h-1.5',
    md: 'w-2 h-2',
    lg: 'w-2.5 h-2.5'
  };

  return (
    <motion.div 
      className={`flex items-center gap-2 rounded-full ${risk.bgColor} ${sizeClasses[size]}`}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <motion.div 
        className={`rounded-full ${risk.dotColor} ${dotSizeClasses[size]}`}
        animate={{ scale: [1, 1.2, 1] }}
        transition={{ duration: 2, repeat: Infinity }}
      />
      <span className={`font-bold ${risk.textColor} tracking-wide`}>
        {showLabel ? `${risk.label} (${formatScore(score || 0)})` : formatScore(score || 0)}
      </span>
    </motion.div>
  );
}
