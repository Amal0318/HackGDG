import { motion } from 'framer-motion';

interface StatusDotProps {
  status: 'connected' | 'degraded' | 'offline';
  label?: string;
  showPulse?: boolean;
}

export default function StatusDot({ status, label, showPulse = true }: StatusDotProps) {
  const colors = {
    connected: 'bg-green-500',
    degraded: 'bg-yellow-500',
    offline: 'bg-red-500'
  };

  const labels = {
    connected: 'Live',
    degraded: 'Degraded',
    offline: 'Offline'
  };

  return (
    <div className="flex items-center gap-2">
      <div className="relative">
        <div className={`w-2 h-2 rounded-full ${colors[status]}`} />
        {showPulse && status !== 'offline' && (
          <motion.div
            className={`absolute top-0 left-0 w-2 h-2 rounded-full ${colors[status]} opacity-75`}
            animate={{ scale: [1, 2, 2], opacity: [0.75, 0, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeOut" }}
          />
        )}
      </div>
      {label !== undefined && (
        <span className="text-xs text-gray-600 font-medium">
          {label || labels[status]}
        </span>
      )}
    </div>
  );
}
