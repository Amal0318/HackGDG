import { motion } from 'framer-motion';
import { AlertCircle } from 'lucide-react';
import clsx from 'clsx';
import VitalSign from './VitalSign';
import RiskBadge from './RiskBadge';

interface PatientCardProps {
  patient: {
    patient_id: string;
    name: string;
    bed_number: string;
    latest_risk_score: number;
    abnormal_vitals?: Array<{ vital: string; value: number; unit: string }>;
    vitals: {
      heart_rate: number;
      systolic_bp: number;
      diastolic_bp: number;
      spo2: number;
      respiratory_rate: number;
      temperature: number;
    };
  };
  onClick: () => void;
}

const PatientCard: React.FC<PatientCardProps> = ({ patient, onClick }) => {
  const abnormalCount = patient.abnormal_vitals?.length || 0;
  const isHighRisk = patient.latest_risk_score >= 70;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      onClick={onClick}
      className={clsx(
        'relative bg-white rounded-lg shadow-md p-4 cursor-pointer transition-all duration-200',
        'hover:scale-[1.02] hover:shadow-xl hover:border-2 hover:border-primary',
        isHighRisk && 'border-2 border-red-300',
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
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <p className="font-bold text-lg text-gray-800">{patient.name}</p>
          <p className="text-sm text-gray-500">Bed {patient.bed_number}</p>
        </div>
        <RiskBadge score={patient.latest_risk_score} />
      </div>

      {/* Vitals Grid */}
      <div className="mt-4 grid grid-cols-3 gap-x-4 gap-y-2 text-sm">
        <VitalSign label="HR" value={patient.vitals.heart_rate} unit="bpm" />
        <VitalSign label="BP" value={`${patient.vitals.systolic_bp}/${patient.vitals.diastolic_bp}`} unit="mmHg" />
        <VitalSign label="RR" value={patient.vitals.respiratory_rate} unit="br/min" />
        <VitalSign label="Temp" value={patient.vitals.temperature.toFixed(1)} unit="°C" />
        <VitalSign label="SpO2" value={patient.vitals.spo2} unit="%" />
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between pt-3 border-t border-gray-200">
        <div className="flex items-center gap-2">
          {abnormalCount > 0 && (
            <div className="flex items-center gap-1 px-2 py-1 bg-red-100 rounded-full">
              <AlertCircle className="w-3 h-3 text-red-600" />
              <span className="text-xs font-semibold text-red-700">{abnormalCount} abnormal</span>
            </div>
          )}
        </div>
        <span className="text-xs text-gray-500">
          Live
        </span>
      </div>
    </motion.div>
  );
};

export default PatientCard;
