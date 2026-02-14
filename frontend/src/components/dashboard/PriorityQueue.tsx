import { useStore } from '../../store';
import { VitalMessage } from '../../types';

interface PriorityQueueProps {
  onPatientSelect: (patientId: string) => void;
}

export const PriorityQueue = ({ onPatientSelect }: PriorityQueueProps) => {
  const patients = useStore(state => state.patients);
  
  // Build priority list
  const priorityPatients = Array.from(patients.entries())
    .map(([patientId, data]) => ({
      patientId,
      latest: data.latest,
      priority: calculatePriority(data.latest)
    }))
    .filter(p => p.latest !== null)
    .sort((a, b) => b.priority - a.priority)
    .slice(0, 5); // Top 5 priority patients

  if (priorityPatients.length === 0) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
        <h3 className="text-gray-900 font-semibold mb-3">🚨 Priority Queue</h3>
        <p className="text-gray-500 text-sm text-center py-4">No patients require attention</p>
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 shadow-sm">
      <h3 className="text-gray-900 font-semibold mb-3">🚨 Priority Queue</h3>
      <div className="space-y-2">
        {priorityPatients.map(({ patientId, latest, priority }) => (
          <button
            key={patientId}
            onClick={() => onPatientSelect(patientId)}
            className={`w-full text-left p-3 rounded-lg border-l-4 transition-all hover:shadow-md ${getPriorityStyles(priority)}`}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="font-bold text-gray-900">{patientId}</p>
                <p className="text-xs text-gray-600 mt-1">
                  {getPriorityReason(latest!)}
                </p>
              </div>
              <div className="text-right">
                <PriorityBadge priority={priority} />
                <p className="text-xs text-gray-500 mt-1">
                  {new Date(latest!.timestamp).toLocaleTimeString()}
                </p>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

// Calculate priority score (higher = more urgent)
function calculatePriority(vitals: VitalMessage | null): number {
  if (!vitals) return 0;
  
  let score = 0;
  
  // State-based priority
  if (vitals.state === 'CRITICAL') score += 100;
  if (vitals.state === 'INTERVENTION') score += 90;
  if (vitals.state === 'EARLY_DETERIORATION') score += 50;
  
  // Event-based priority
  if (vitals.event_type === 'SEPSIS_ALERT') score += 80;
  if (vitals.event_type === 'HYPOTENSION') score += 60;
  if (vitals.event_type === 'HYPOXIA') score += 70;
  if (vitals.event_type === 'TACHYCARDIA') score += 40;
  
  // Risk-based priority
  if (vitals.risk_score !== undefined) {
    score += vitals.risk_score;
  }
  
  // Anomaly bonus
  if (vitals.anomaly_detected) score += 30;
  
  return score;
}

function getPriorityStyles(priority: number): string {
  if (priority >= 150) return 'border-red-500 bg-red-50 hover:bg-red-100';
  if (priority >= 100) return 'border-orange-500 bg-orange-50 hover:bg-orange-100';
  if (priority >= 50) return 'border-yellow-500 bg-yellow-50 hover:bg-yellow-100';
  return 'border-blue-500 bg-blue-50 hover:bg-blue-100';
}

function getPriorityReason(vitals: VitalMessage): string {
  const reasons: string[] = [];
  
  if (vitals.state === 'CRITICAL') reasons.push('Critical state');
  if (vitals.state === 'INTERVENTION') reasons.push('Intervention required');
  if (vitals.event_type === 'SEPSIS_ALERT') reasons.push('Sepsis alert');
  if (vitals.event_type === 'HYPOXIA') reasons.push('Low oxygen');
  if (vitals.event_type === 'HYPOTENSION') reasons.push('Low BP');
  if (vitals.anomaly_detected) reasons.push(`Anomaly: ${vitals.anomaly_type}`);
  if (vitals.risk_score && vitals.risk_score >= 70) reasons.push('High risk');
  
  return reasons.join(' • ') || 'Monitoring required';
}

const PriorityBadge = ({ priority }: { priority: number }) => {
  if (priority >= 150) {
    return <span className="bg-red-600 text-white text-xs px-2 py-1 rounded-full font-bold">URGENT</span>;
  }
  if (priority >= 100) {
    return <span className="bg-orange-600 text-white text-xs px-2 py-1 rounded-full font-bold">HIGH</span>;
  }
  if (priority >= 50) {
    return <span className="bg-yellow-600 text-white text-xs px-2 py-1 rounded-full font-bold">MEDIUM</span>;
  }
  return <span className="bg-blue-600 text-white text-xs px-2 py-1 rounded-full font-bold">LOW</span>;
};
