import { useStore } from '../../store';

export const DashboardStats = () => {
  const patients = useStore(state => state.patients);
  
  // Calculate statistics
  const stats = calculateStats(patients);
  
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <StatCard
        label="Total Patients"
        value={stats.total}
        icon="👥"
        color="bg-blue-50 border-blue-200 text-blue-700"
      />
      <StatCard
        label="Critical"
        value={stats.critical}
        icon="🚨"
        color="bg-red-50 border-red-200 text-red-700"
      />
      <StatCard
        label="Anomalies"
        value={stats.anomalies}
        icon="⚠️"
        color="bg-orange-50 border-orange-200 text-orange-700"
      />
      <StatCard
        label="Stable"
        value={stats.stable}
        icon="✅"
        color="bg-green-50 border-green-200 text-green-700"
      />
    </div>
  );
};

const StatCard = ({ label, value, icon, color }: {
  label: string;
  value: number;
  icon: string;
  color: string;
}) => (
  <div className={`${color} border-2 rounded-lg p-4 shadow-sm`}>
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm font-medium opacity-80">{label}</p>
        <p className="text-3xl font-bold mt-1">{value}</p>
      </div>
      <span className="text-4xl opacity-60">{icon}</span>
    </div>
  </div>
);

function calculateStats(patients: Map<string, any>) {
  let total = 0;
  let critical = 0;
  let anomalies = 0;
  let stable = 0;
  
  patients.forEach((data) => {
    if (data.latest) {
      total++;
      
      if (data.latest.state === 'CRITICAL' || data.latest.state === 'INTERVENTION') {
        critical++;
      }
      
      if (data.latest.state === 'STABLE') {
        stable++;
      }
      
      if (data.latest.anomaly_detected) {
        anomalies++;
      }
    }
  });
  
  return { total, critical, anomalies, stable };
}
