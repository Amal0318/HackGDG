import { motion, AnimatePresence } from 'framer-motion';
import { X, AlertTriangle, Info, AlertCircle } from 'lucide-react';
import { useEffect } from 'react';
import clsx from 'clsx';

interface AlertBannerProps {
  severity: 'critical' | 'warning' | 'info';
  patientName: string;
  message: string;
  onDismiss: () => void;
  onView?: () => void;
  autoDismiss?: boolean;
}

export default function AlertBanner({
  severity,
  patientName,
  message,
  onDismiss,
  onView,
  autoDismiss = false
}: AlertBannerProps) {
  useEffect(() => {
    if (autoDismiss && severity !== 'critical') {
      const timer = setTimeout(onDismiss, 10000);
      return () => clearTimeout(timer);
    }
  }, [autoDismiss, severity, onDismiss]);

  const styles = {
    critical: {
      bg: 'bg-red-100 border-red-500',
      text: 'text-red-900',
      icon: AlertCircle,
      iconColor: 'text-red-600'
    },
    warning: {
      bg: 'bg-amber-100 border-amber-500',
      text: 'text-amber-900',
      icon: AlertTriangle,
      iconColor: 'text-amber-600'
    },
    info: {
      bg: 'bg-blue-100 border-blue-500',
      text: 'text-blue-900',
      icon: Info,
      iconColor: 'text-blue-600'
    }
  };

  const style = styles[severity];
  const Icon = style.icon;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1, x: severity === 'critical' ? [0, -2, 2, -2, 2, 0] : 0 }}
        exit={{ y: -100, opacity: 0 }}
        transition={{ duration: 0.3 }}
        className={clsx(
          'border-l-4 p-4 rounded-lg shadow-lg',
          style.bg
        )}
      >
        <div className="flex items-start gap-3">
          <Icon className={clsx('w-5 h-5 mt-0.5 flex-shrink-0', style.iconColor)} />
          <div className="flex-1 min-w-0">
            <p className={clsx('font-semibold text-sm', style.text)}>
              {patientName}
            </p>
            <p className={clsx('text-sm mt-1', style.text)}>
              {message}
            </p>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            {onView && (
              <button
                onClick={onView}
                className={clsx(
                  'px-3 py-1 text-xs font-medium rounded-md hover:opacity-80 transition-opacity',
                  severity === 'critical' ? 'bg-red-600 text-white' :
                  severity === 'warning' ? 'bg-amber-600 text-white' :
                  'bg-blue-600 text-white'
                )}
              >
                View Patient
              </button>
            )}
            <button
              onClick={onDismiss}
              className={clsx('p-1 rounded-md hover:bg-black/10 transition-colors', style.text)}
              aria-label="Dismiss"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}
