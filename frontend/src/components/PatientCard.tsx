import { motion } from 'framer-motion';
import { AlertCircle, CheckCircle, Eye } from 'lucide-react';
import clsx from 'clsx';
import VitalSign from './VitalSign';
import RiskBadge from './RiskBadge';
import { useState } from 'react';
import { alertAPI } from '../services/api';

interface PatientCardProps {
  patient: {
    patient_id: string;
    name: string;
    bed_number: string;
    latest_risk_score: number;
    abnormal_vitals?: Array<{ vital: string; value: number; unit: string }>;
    alert_acknowledged?: boolean;
    acknowledged_by?: string;
    vitals: {
      heart_rate: number;
      systolic_bp: number;
      diastolic_bp: number;
      spo2: number;
      respiratory_rate: number;
      temperature: number;
      lactate?: number;
      shock_index?: number;
    };
  };
  onClick: () => void;
  onAcknowledge?: (patientId: string) => void;
}

const PatientCard: React.FC<PatientCardProps> = ({ patient, onClick, onAcknowledge }) => {
  const abnormalCount = patient.abnormal_vitals?.length || 0;
  const isHighRisk = patient.latest_risk_score >= 70;
  const [acknowledging, setAcknowledging] = useState(false);

  const handleAcknowledge = async (e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent card click
    setAcknowledging(true);
    
    try {
      await alertAPI.acknowledgeAlert(patient.patient_id, "Dr. Smith");
      if (onAcknowledge) {
        onAcknowledge(patient.patient_id);
      }
    } catch (error) {
      console.error("Failed to acknowledge alert:", error);
    } finally {
      setAcknowledging(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      onClick={onClick}
      className={clsx(
        'relative bg-white rounded-lg shadow-md cursor-pointer transition-all duration-200',
        'hover:scale-[1.02] hover:shadow-xl hover:border-2 hover:border-primary',
        isHighRisk && !patient.alert_acknowledged && 'border-2 border-red-300 animate-pulse',
        isHighRisk && patient.alert_acknowledged && 'border-2 border-green-300',
        'focus-visible-ring overflow-hidden'
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
      {/* High Risk Alert Banner */}
      {isHighRisk && !patient.alert_acknowledged && (
        <div className="bg-red-500 text-white text-xs font-bold px-3 py-2 flex items-center justify-between">
          <span className="flex items-center gap-1">
            <AlertCircle className="w-3 h-3" />
            HIGH RISK ALERT
          </span>
          <button
            onClick={handleAcknowledge}
            disabled={acknowledging}
            className="bg-white text-red-600 px-2 py-1 rounded text-xs font-semibold hover:bg-red-50 transition-colors flex items-center gap-1 disabled:opacity-50"
          >
            <Eye className="w-3 h-3" />
            {acknowledging ? "..." : "Acknowledge"}
          </button>
        </div>
      )}

      {/* Acknowledged Status */}
      {isHighRisk && patient.alert_acknowledged && (
        <div className="bg-green-500 text-white text-xs font-bold px-3 py-2 flex items-center gap-1">
          <CheckCircle className="w-3 h-3" />
          Alert Acknowledged {patient.acknowledged_by ? `by ${patient.acknowledged_by}` : ''}
        </div>
      )}

      {/* Card Content */}
      <div className="p-4">
        {/* Header */}
        <div className="flex justify-between items-start mb-4">
          <div className="min-w-0 flex-1">
            <p className="font-bold text-lg text-gray-800 truncate">{patient.name}</p>
            <p className="text-sm text-gray-500">Bed {patient.bed_number}</p>
          </div>
          <div className="ml-3 flex-shrink-0">
            <RiskBadge score={patient.latest_risk_score} />
          </div>
        </div>

        {/* Vitals Grid */}
        <div className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm mb-4">
          <VitalSign label="HR" value={patient.vitals.heart_rate} unit="bpm" />
          <VitalSign label="BP" value={`${patient.vitals.systolic_bp}/${patient.vitals.diastolic_bp}`} unit="mmHg" />
          <VitalSign label="SpO2" value={patient.vitals.spo2} unit="%" />
          <VitalSign label="RR" value={patient.vitals.respiratory_rate} unit="br/min" />
          <VitalSign label="Temp" value={patient.vitals.temperature.toFixed(1)} unit="°C" />
          <VitalSign 
            label="Lactate" 
            value={patient.vitals.lactate?.toFixed(2) || 'N/A'} 
            unit="mmol/L"
            status={patient.vitals.lactate && patient.vitals.lactate > 2.0 ? 'warning' : 'normal'}
          />
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between pt-3 border-t border-gray-200">
          <div className="flex items-center gap-2 min-w-0">
            {abnormalCount > 0 && (
              <div className="flex items-center gap-1 px-2 py-1 bg-red-100 rounded-full flex-shrink-0">
                <AlertCircle className="w-3 h-3 text-red-600" />
                <span className="text-xs font-semibold text-red-700 whitespace-nowrap">{abnormalCount} abnormal</span>
              </div>
            )}
          </div>
          <span className="text-xs text-gray-500 flex-shrink-0 ml-2">
            Live
          </span>
        </div>
      </div>
    </motion.div>
  );
};

export default PatientCard;
