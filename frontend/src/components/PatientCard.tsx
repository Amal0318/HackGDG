import { motion } from 'framer-motion';
import { AlertCircle, AlertTriangle } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import clsx from 'clsx';
import VitalSign from './VitalSign';
import RiskBadge from './RiskBadge';
import { Patient } from '../mockData';
import { getVitalStatus } from '../utils/calculations';

interface PatientCardProps {
  patient: Patient;
  onClick: () => void;
  isAssigned?: boolean;
  index?: number;
}

export default function PatientCard({ patient, onClick, isAssigned, index = 0 }: PatientCardProps) {
  const hasActiveAlerts = patient.alerts.length > 0;
  const criticalAlerts = patient.alerts.filter(a => a.severity === 'critical').length;
  const warningAlerts = patient.alerts.filter(a => a.severity === 'warning').length;

  const getTrend = (current: number, history: number[]): 'up' | 'down' | 'stable' => {
    const recent = history.slice(-5);
    const avg = recent.reduce((a, b) => a + b, 0) / recent.length;
    if (current > avg + 2) return 'up';
    if (current < avg - 2) return 'down';
    return 'stable';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.05 }}
      onClick={onClick}
      className={clsx(
        'relative bg-white rounded-lg shadow-md p-4 cursor-pointer transition-all duration-200',
        'hover:scale-[1.02] hover:shadow-xl hover:border-2 hover:border-primary',
        hasActiveAlerts && patient.riskLevel === 'critical' && 'pulse-border border-2',
        'focus-visible-ring'
      )}
      tabIndex={0}
      role="button"
      aria-label={`View details for ${patient.name}`}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      }}
    >
      {/* Bookmark for assigned patients */}
      {isAssigned && (
        <div className="absolute top-0 right-4 w-0 h-0 border-l-[12px] border-l-transparent border-r-[12px] border-r-transparent border-t-[16px] border-t-primary"></div>
      )}

      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1 min-w-0">
          <h3 className="font-bold text-lg text-gray-900 truncate">
            {patient.bed}
          </h3>
          <p className="text-sm text-gray-600 truncate">{patient.name}</p>
          <p className="text-xs text-gray-500">{patient.age} years</p>
        </div>
        <div className="flex-shrink-0 ml-2">
          <RiskBadge score={patient.riskScore} size="sm" showLabel={false} />
        </div>
      </div>

      {/* Vitals Grid */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        <VitalSign
          label="HR"
          value={patient.vitals.heartRate}
          unit="bpm"
          trend={getTrend(patient.vitals.heartRate, patient.trends.heartRate)}
          status={getVitalStatus(patient.vitals.heartRate, [60, 100])}
        />
        <VitalSign
          label="BP"
          value={`${patient.vitals.systolicBP}/${patient.vitals.diastolicBP}`}
          unit="mmHg"
          status={getVitalStatus(patient.vitals.systolicBP, [90, 140])}
        />
        <VitalSign
          label="SpO2"
          value={patient.vitals.spo2}
          unit="%"
          trend={getTrend(patient.vitals.spo2, patient.trends.spo2)}
          status={getVitalStatus(patient.vitals.spo2, [95, 100])}
        />
        <VitalSign
          label="RR"
          value={patient.vitals.respiratoryRate}
          unit="/min"
          trend={getTrend(patient.vitals.respiratoryRate, patient.trends.respiratoryRate)}
          status={getVitalStatus(patient.vitals.respiratoryRate, [12, 20])}
        />
        <VitalSign
          label="Temp"
          value={patient.vitals.temperature}
          unit="°C"
          status={getVitalStatus(patient.vitals.temperature, [36.5, 37.5])}
        />
        <VitalSign
          label="GCS"
          value={patient.vitals.gcs}
          unit=""
          status={patient.vitals.gcs === 15 ? 'normal' : patient.vitals.gcs >= 13 ? 'warning' : 'critical'}
        />
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between pt-3 border-t border-gray-200">
        <div className="flex items-center gap-2">
          {criticalAlerts > 0 && (
            <div className="flex items-center gap-1 px-2 py-1 bg-red-100 rounded-full">
              <AlertCircle className="w-3 h-3 text-red-600" />
              <span className="text-xs font-semibold text-red-700">{criticalAlerts}</span>
            </div>
          )}
          {warningAlerts > 0 && (
            <div className="flex items-center gap-1 px-2 py-1 bg-amber-100 rounded-full">
              <AlertTriangle className="w-3 h-3 text-amber-600" />
              <span className="text-xs font-semibold text-amber-700">{warningAlerts}</span>
            </div>
          )}
        </div>
        <span className="text-xs text-gray-500">
          Updated {formatDistanceToNow(new Date(), { addSuffix: true })}
        </span>
      </div>
    </motion.div>
  );
}
